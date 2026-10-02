#!/usr/bin/env python3
"""
Interview Agent
================
Conducts a structured mock interview: generates role-specific questions,
scores the candidate's answers as they're given, and produces a final
evaluation with strengths and gaps.

Usage:
    python interview_agent.py
    python interview_agent.py --role "Backend Engineer (Python)" --num-questions 6
    python interview_agent.py --role "Data Analyst" --answers-file sample_answers.txt

See README.md for full setup instructions.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Pick a provider with the API_PROVIDER env var: "anthropic" (default),
# "openai", or "groq" (groq is free and OpenAI-SDK compatible).
# Each provider needs its own API key env var (see README.md).

PROVIDER = os.environ.get("API_PROVIDER", "anthropic").lower()

PROVIDER_DEFAULTS = {
    "anthropic": {"key_env": "ANTHROPIC_API_KEY", "model": "claude-sonnet-5", "base_url": None},
    "openai":    {"key_env": "OPENAI_API_KEY",    "model": "gpt-4o-mini",      "base_url": None},
    "groq":      {"key_env": "GROQ_API_KEY",       "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
}

if PROVIDER not in PROVIDER_DEFAULTS:
    print(f"Unknown API_PROVIDER '{PROVIDER}'. Choose one of: {', '.join(PROVIDER_DEFAULTS)}")
    sys.exit(1)

DEFAULT_MODEL = os.environ.get("INTERVIEW_AGENT_MODEL", PROVIDER_DEFAULTS[PROVIDER]["model"])
TRANSCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")

SYSTEM_QUESTION_GEN = """You are an experienced technical/behavioral interviewer.
Given a target role or skill area, generate a numbered list of interview questions
that a strong interviewer would ask a candidate for that role.

Rules:
- Mix question types: a couple of conceptual/technical questions, at least one
  practical/scenario question, and at least one behavioral question, unless the
  role clearly calls for something else.
- Questions should increase roughly in difficulty.
- Each question should be answerable in a few sentences to a short paragraph
  (no essay-length prompts).
- Return ONLY valid JSON: a list of strings. No preamble, no markdown fences.
"""

SYSTEM_SCORER = """You are a strict but fair technical interviewer scoring a
candidate's answer to a single interview question.

Score the answer from 1 to 10, where:
1-3  = missing or fundamentally incorrect / no real understanding shown
4-6  = partially correct, missing key points or lacking depth
7-8  = solid, correct, reasonably complete answer
9-10 = excellent, precise, shows depth and real-world judgement

Return ONLY valid JSON with this exact shape, no markdown fences, no preamble:
{"score": <integer 1-10>, "feedback": "<one or two sentence rationale>"}
"""

SYSTEM_FINAL_EVAL = """You are compiling the final evaluation for a completed mock
interview, based on the full transcript of questions, answers, and per-question
scores.

Return ONLY valid JSON with this exact shape, no markdown fences, no preamble:
{
  "overall_score": <float, 1-10, weighted average impression not just a mean>,
  "recommendation": "<one of: Strong Hire, Hire, Leaning Hire, Leaning No Hire, No Hire>",
  "strengths": ["<short bullet>", "..."],
  "gaps": ["<short bullet>", "..."],
  "summary": "<2-4 sentence overall narrative summary>"
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str):
    """Strip markdown code fences if present and parse JSON."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return json.loads(text.strip())


class InterviewAgent:
    def __init__(self, model: str = DEFAULT_MODEL, provider: str = PROVIDER):
        self.provider = provider
        cfg = PROVIDER_DEFAULTS[provider]
        api_key = os.environ.get(cfg["key_env"])
        if not api_key:
            print(f"ERROR: {cfg['key_env']} environment variable is not set.")
            print(f"(Using provider '{provider}' -- set API_PROVIDER to switch. See README.md.)")
            sys.exit(1)

        if provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                print("Missing dependency. Run: pip install -r requirements.txt")
                sys.exit(1)
            self.client = Anthropic(api_key=api_key)
        else:
            # Both "openai" and "groq" use the OpenAI SDK; groq just points
            # at a different base_url since it's OpenAI-API-compatible.
            try:
                from openai import OpenAI
            except ImportError:
                print("Missing dependency. Run: pip install -r requirements.txt")
                sys.exit(1)
            self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

        self.model = model

    def _call(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(block.text for block in response.content if block.type == "text")
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""

    # -- Step 1: generate questions -----------------------------------
    def generate_questions(self, role: str, num_questions: int = 5) -> list:
        prompt = f'Target role / skill area: "{role}"\nGenerate exactly {num_questions} interview questions.'
        raw = self._call(SYSTEM_QUESTION_GEN, prompt)
        try:
            questions = extract_json(raw)
            assert isinstance(questions, list)
            return [str(q) for q in questions[:num_questions]]
        except (json.JSONDecodeError, AssertionError):
            # Fallback: split on numbered lines if the model didn't return clean JSON
            lines = [re.sub(r"^\d+[\.\)]\s*", "", l).strip() for l in raw.splitlines() if l.strip()]
            return lines[:num_questions] if lines else [f"Tell me about your experience with {role}."]

    # -- Step 2: score a single answer ----------------------------------
    def score_answer(self, role: str, question: str, answer: str) -> dict:
        prompt = (
            f'Role: "{role}"\n'
            f'Question: "{question}"\n'
            f'Candidate answer: "{answer}"\n\n'
            "Score this answer."
        )
        raw = self._call(SYSTEM_SCORER, prompt, max_tokens=300)
        try:
            result = extract_json(raw)
            result["score"] = max(1, min(10, int(result["score"])))
            return result
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return {"score": 5, "feedback": "Could not automatically parse a score; defaulted to 5."}

    # -- Step 3: final evaluation ----------------------------------------
    def final_evaluation(self, role: str, transcript: list) -> dict:
        prompt = f'Role: "{role}"\nFull transcript (JSON):\n{json.dumps(transcript, indent=2)}'
        raw = self._call(SYSTEM_FINAL_EVAL, prompt, max_tokens=800)
        try:
            return extract_json(raw)
        except json.JSONDecodeError:
            avg = sum(t["score"] for t in transcript) / len(transcript)
            return {
                "overall_score": round(avg, 1),
                "recommendation": "Leaning Hire" if avg >= 6 else "Leaning No Hire",
                "strengths": [],
                "gaps": [],
                "summary": "Automated summary generation failed; showing computed average score only.",
            }


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_session(role: str, num_questions: int, answers_file: str = None, questions_file: str = None):
    agent = InterviewAgent()

    print(f"\n=== Interview Agent ===")
    print(f"Role: {role}")

    if questions_file:
        with open(questions_file, "r") as f:
            questions = [line.strip() for line in f if line.strip()][:num_questions]
        print(f"Using {len(questions)} questions from {questions_file}\n")
    else:
        print(f"Generating {num_questions} interview questions...\n")
        questions = agent.generate_questions(role, num_questions)

    # Pre-supplied answers (for scripted/demo runs) or interactive typing
    canned_answers = None
    if answers_file:
        with open(answers_file, "r") as f:
            canned_answers = [line.strip() for line in f if line.strip()]

    transcript = []
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        if canned_answers and i - 1 < len(canned_answers):
            answer = canned_answers[i - 1]
            print(f"(using provided answer): {answer}")
        else:
            answer = input("Your answer: ").strip()
            if not answer:
                answer = "(no answer given)"

        result = agent.score_answer(role, question, answer)
        print(f"  -> Score: {result['score']}/10 — {result['feedback']}\n")

        transcript.append({
            "question_number": i,
            "question": question,
            "answer": answer,
            "score": result["score"],
            "feedback": result["feedback"],
        })

    print("Compiling final evaluation...\n")
    evaluation = agent.final_evaluation(role, transcript)

    session = {
        "role": role,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": agent.model,
        "transcript": transcript,
        "final_evaluation": evaluation,
    }

    print_summary(session)
    save_session(session)
    return session


def print_summary(session: dict):
    ev = session["final_evaluation"]
    print("=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    print(f"Overall score: {ev.get('overall_score')}/10")
    print(f"Recommendation: {ev.get('recommendation')}")
    print("\nStrengths:")
    for s in ev.get("strengths", []):
        print(f"  + {s}")
    print("\nGaps:")
    for g in ev.get("gaps", []):
        print(f"  - {g}")
    print(f"\nSummary: {ev.get('summary')}")
    print("=" * 50)


def save_session(session: dict):
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_role = re.sub(r"[^a-zA-Z0-9]+", "_", session["role"]).strip("_").lower()

    json_path = os.path.join(TRANSCRIPT_DIR, f"{ts}_{safe_role}.json")
    with open(json_path, "w") as f:
        json.dump(session, f, indent=2)

    txt_path = os.path.join(TRANSCRIPT_DIR, f"{ts}_{safe_role}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Interview transcript — Role: {session['role']}\n")
        f.write(f"Timestamp: {session['timestamp']}\n")
        f.write(f"Model: {session['model']}\n\n")
        for t in session["transcript"]:
            f.write(f"Q{t['question_number']}: {t['question']}\n")
            f.write(f"A: {t['answer']}\n")
            f.write(f"Score: {t['score']}/10 — {t['feedback']}\n\n")
        ev = session["final_evaluation"]
        f.write("=" * 50 + "\n")
        f.write("FINAL EVALUATION\n")
        f.write("=" * 50 + "\n")
        f.write(f"Overall score: {ev.get('overall_score')}/10\n")
        f.write(f"Recommendation: {ev.get('recommendation')}\n")
        f.write("Strengths:\n")
        for s in ev.get("strengths", []):
            f.write(f"  + {s}\n")
        f.write("Gaps:\n")
        for g in ev.get("gaps", []):
            f.write(f"  - {g}\n")
        f.write(f"Summary: {ev.get('summary')}\n")

    print(f"\nSaved transcript to:\n  {json_path}\n  {txt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a structured mock interview session.")
    parser.add_argument("--role", type=str, default=None, help="Target role or skill area to interview for.")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to ask (default: 5).")
    parser.add_argument("--answers-file", type=str, default=None,
                         help="Optional path to a text file with one answer per line, for scripted/demo runs.")
    parser.add_argument("--questions-file", type=str, default=None,
                         help="Optional path to a text file with one question per line. "
                              "Skips generation so runs are reproducible and match a given --answers-file.")
    args = parser.parse_args()

    role = args.role or input("What role or skill area is this interview for? ").strip()
    if not role:
        print("A role is required.")
        sys.exit(1)

    num_questions = max(5, args.num_questions)  # spec requires at least 5

    run_session(role, num_questions, args.answers_file, args.questions_file)


if __name__ == "__main__":
    main()

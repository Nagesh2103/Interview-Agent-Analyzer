# Interview Agent

A command-line agent that conducts a structured mock interview: it generates
role-specific questions, scores each answer as you give it, and produces a
final evaluation with strengths and gaps.

**One sentence:** My agent takes a target role (e.g. "Data Analyst") and a
candidate's typed answers, and produces per-question scores plus a final
hire/no-hire style evaluation with strengths and gaps.

---

## What it does

1. Generates at least 5 role-specific interview questions (mix of technical,
   scenario, and behavioral).
2. Asks each question and accepts a typed answer.
3. Scores each answer 1–10 with a short rationale, using the LLM as judge.
4. Produces a final evaluation: overall score, hire recommendation,
   strengths, and gaps.
5. Saves the full transcript to `transcripts/` as both `.json` and `.txt`.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Pick a provider and get an API key

This agent supports three providers -- pick whichever you have access to:

| Provider | Sign up | Notes |
|---|---|---|
| `anthropic` (default) | console.anthropic.com | Requires purchasing credits |
| `openai` | platform.openai.com | Requires purchasing credits |
| `groq` | console.groq.com | Free tier, no card required to start |

### 3. Set your API key (and provider, if not Anthropic)

Windows (cmd):
```cmd
set API_PROVIDER=groq
set GROQ_API_KEY=gsk-your-key-here
```

macOS/Linux:
```bash
export API_PROVIDER=groq
export GROQ_API_KEY=gsk-your-key-here
```

If you're using the default (anthropic), you can skip setting
API_PROVIDER and just set ANTHROPIC_API_KEY.

(Or copy `.env.example` to `.env`, fill it in, and load it with your shell /
`direnv` / `python-dotenv` of choice.)

---

## Running it

### Interactive mode (type your own answers)

```bash
python interview_agent.py --role "Data Analyst"
```

You'll be prompted with each question and type your answer at the terminal.

### Scripted/demo mode (no typing needed, fully reproducible)

```bash
python interview_agent.py --role "Data Analyst" --questions-file sample_questions.txt --answers-file sample_answers.txt
```

`--questions-file` locks in a fixed set of questions instead of generating
new ones each run. This matters because question generation is
non-deterministic -- the same `--role` can produce different questions on
different runs or providers, which would make a fixed `--answers-file`
mismatch what's actually being asked. Passing both files together gives an
identical, reproducible run every time, regardless of provider.

You can also use `--answers-file` alone (with freshly generated questions)
for a quicker, non-reproducible scripted run -- answers are matched to
questions by position, so they'll only make sense if they happen to align.

### Options

| Flag | Description | Default |
|---|---|---|
| `--role` | Target role or skill area | prompted interactively if omitted |
| `--num-questions` | Number of questions to ask (minimum 5, enforced) | 5 |
| `--questions-file` | Path to a text file with one question per line, to skip generation for reproducible runs | none (generates fresh questions) |
| `--answers-file` | Path to a text file with one answer per line, for scripted runs | none (interactive typing) |

### Example

```bash
python interview_agent.py --role "Backend Engineer (Python)" --num-questions 6
```

---

## Output

Every run prints a live transcript to the terminal and saves two files to
`transcripts/`:

- `<timestamp>_<role>.json` — structured data (questions, answers, scores,
  feedback, final evaluation) for programmatic use.
- `<timestamp>_<role>.txt` — human-readable transcript, same content.

A pre-generated example is included at [`sample_transcript.txt`](sample_transcript.txt)
so reviewers can see the expected format without needing an API key.

---

## Design & architecture

The agent follows a simple Input → Think → Act → Output loop, repeated per
question, plus one final aggregation step:

```
role input
   │
   ▼
generate_questions()  ──▶  LLM call, structured JSON output
   │
   ▼
for each question:
   ask candidate  ──▶  score_answer()  ──▶  LLM call, structured JSON output
   │
   ▼
final_evaluation()  ──▶  LLM call over full transcript, structured JSON output
   │
   ▼
print + save transcript (.json + .txt)
```

Three separate system prompts (question generation, per-answer scoring,
final evaluation) rather than one big prompt, so each LLM call has a single,
narrow job and returns a small, easy-to-validate JSON shape.

---

## Provider support

The agent is provider-agnostic: `API_PROVIDER` selects between `anthropic`
(default), `openai`, and `groq`. `openai` and `groq` share one code path
since Groq exposes an OpenAI-compatible API -- only the `base_url` differs.
Swapping providers requires no code changes, only environment variables.

## Design tradeoffs & limitations

- **No resume/job-description ingestion.** Questions are generated purely
  from the `--role` string. Feeding in a real job description or resume
  (via a PDF/text file, same pattern as the `--answers-file` flag) would
  make questions more targeted — left out to keep the 24-hour scope
  manageable.
- **No audio transcription.** The spec allows "typed or transcribed"
  answers; this implementation only accepts typed/text answers (including
  from a file, so a transcript from another tool could be piped in via
  `--answers-file`). Wiring in real speech-to-text was out of scope.
- **Per-answer scoring is independent of later answers.** Each answer is
  scored in isolation before the next question is asked, rather than
  waiting to score everything at the end. This keeps the loop simple and
  gives the candidate/interviewer live feedback, at the cost of not letting
  earlier scores be revised in light of later answers.
- **No persistent question bank / caching.** Questions are regenerated by
  the LLM on every run rather than pulled from a stored bank. This keeps
  the code simple and always gives fresh, role-tailored questions, at the
  cost of an extra API call and less repeatability across runs for the same
  role.
- **JSON parsing has fallbacks, not full validation.** LLM responses are
  expected to be JSON; if parsing fails, the code falls back to
  best-effort parsing (question generation) or a neutral default score of 5
  (answer scoring) rather than crashing, since a live demo failing on a
  single malformed response would be worse than a slightly degraded
  fallback.
- **Single-file implementation.** Everything lives in `interview_agent.py`
  for readability within the 24-hour scope, rather than split into
  separate modules for question generation, scoring, and persistence.

---

## Repository structure

```
interview-agent/
├── interview_agent.py     # main agent
├── requirements.txt       # dependencies
├── .env.example           # example environment file
├── sample_answers.txt     # canned answers for scripted demo runs
├── sample_transcript.txt  # example output, pre-generated
├── transcripts/           # created at runtime, holds real run outputs
└── README.md
```

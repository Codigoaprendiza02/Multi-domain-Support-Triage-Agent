# HackerRank Orchestrate

This repository is my submission for the **HackerRank Orchestrate** 24-hour hackathon (May 1–2, 2026).

I built a terminal-based support triage agent that reads tickets across three ecosystems - **HackerRank**, **Claude**, and **Visa** - and produces grounded, safe routing decisions using only the support corpus in this repo.

## What the problem statement was

The challenge was to take the provided `support_tickets.csv` input and generate a completed `output.csv` for every row. The agent had to:

- identify the request type
- classify the issue into a product area
- assess urgency and risk
- decide whether to answer or escalate
- retrieve the most relevant support documentation
- generate a safe, corpus-grounded response

The solution also had to stay terminal-based, avoid hallucinations, use only the provided support corpus, and escalate sensitive or unsupported cases instead of guessing. The full statement is in [problem_statement.md](problem_statement.md).

## My approach

I implemented the agent as a local-first pipeline with safety and retrieval built in:

1. I ingest and chunk the local support corpus.
2. I retrieve context with a hybrid retrieval pipeline and reranking.
3. I run security checks for prompt injection, PII, and escalation triggers.
4. I ask the LLM for a structured answer when available.
5. I fall back to deterministic local behavior when provider quota, rate limits, or parsing issues block live generation.
6. I write the final structured rows to `support_tickets/output.csv` and log each turn to the shared transcript file.

I also hardened the runtime around provider quota issues, shortened prompts, added cooldown/backoff logic, and kept the output schema consistent with the grader requirements.

## What I did

- Built the triage pipeline in `code/`.
- Added retrieval, security, parsing, and sanitization layers.
- Wired structured output fields including `status`, `product_area`, `response`, `resolution`, `justification`, and `request_type`.
- Added Docker support for reproducible local runs.
- Wrote tests for the pipeline phases and the final end-to-end flow.
- Kept the README files and submission checklist aligned with the final package.

## Setup instructions

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` at the repository root and set your API keys if you want live provider calls.

```env
GEMINI_API_KEY=
OPENAI_API_KEY=
```

I kept secrets out of source control and read them only from environment variables.

### 3. Run the agent

```powershell
.\.venv\Scripts\python.exe code\main.py
```

For a sample run:

```powershell
.\.venv\Scripts\python.exe code\main.py --sample
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python.exe code\main.py --dry-run
```

### 4. Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest code\tests -q
```

### 5. Build and run with Docker

```powershell
docker build -t support-triage-agent ./code
docker run --rm --env-file .env -v ${PWD}/data:/data:ro -v ${PWD}/support_tickets:/support_tickets support-triage-agent python main.py --sample
```

## Output contract

My agent writes `support_tickets/output.csv` with these fields:

| Column | Meaning |
| --- | --- |
| `ticket_id` | input ticket identifier |
| `status` | `answered` or `escalated` |
| `product_area` | most relevant support domain |
| `response` | user-facing answer grounded in the corpus |
| `resolution` | short routing note such as `see response` or `escalated:security` |
| `justification` | concise explanation of the decision |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

## Repository layout

- `code/` - the agent implementation
- `data/` - the local support corpus
- `support_tickets/` - input and output CSV files
- `FINAL_SUBMISSION_CHECKLIST.md` - my final pre-submit gate

## Chat transcript logging

I used the shared transcript log required by `AGENTS.md`:

- Windows: `%USERPROFILE%\hackerrank_orchestrate\log.txt`
- macOS/Linux: `$HOME/hackerrank_orchestrate/log.txt`

## Submission

For submission, I packaged the code, the final predictions CSV, and the chat transcript. The evaluation also references [evalutation_criteria.md](evalutation_criteria.md).

## Notes

- I kept the agent deterministic where possible.
- I escalated high-risk or unsupported cases instead of guessing.
- I updated `code/README.md` separately with implementation-level usage details.
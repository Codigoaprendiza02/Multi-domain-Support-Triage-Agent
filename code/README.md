# Support Triage Agent

Production-style, terminal-first support ticket triage system for the HackerRank Orchestrate challenge.

This service processes tickets from `support_tickets/support_tickets.csv`, retrieves grounded context from local corpora (`data/`), applies security and escalation checks, and writes structured decisions to `support_tickets/output.csv`.

## Table of Contents
- Overview
- Core Capabilities
- Repository Contract
- Runtime Requirements
- Configuration
- Quickstart (Local)
- Usage
- Docker Usage
- Architecture
- Quality Gates and Testing
- Output Contract
- Operational Notes
- Troubleshooting

## Overview
The agent performs end-to-end triage with deterministic, local-first defaults and safe fallbacks:
- Ingest and chunk local corpus data.
- Retrieve relevant support context (hybrid retrieval + reranking).
- Run security checks and escalation logic.
- Generate structured response fields using live LLMs when available.
- Fall back gracefully to local deterministic behavior on provider errors/quota issues.

## Core Capabilities
- Multi-domain support (`HackerRank`, `Claude`, `Visa`, and cross-domain tickets).
- Security-first escalation for high-risk scenarios.
- Corpus-grounded responses (no live web grounding).
- Quota-aware LLM behavior (retry controls, cooldown, short outputs).
- Transcript logging to `$HOME/hackerrank_orchestrate/log.txt`.

## Repository Contract
- Entry point: `code/main.py`
- Input CSV: `support_tickets/support_tickets.csv`
- Output CSV: `support_tickets/output.csv`
- Shared transcript log: `%USERPROFILE%/hackerrank_orchestrate/log.txt` (Windows)

## Runtime Requirements
- Python 3.11+
- OS: Windows/macOS/Linux
- Optional Docker Desktop (for containerized runs)

Python dependencies are pinned in `requirements.txt`.

## Configuration
Create a `.env` file at repository root (not inside `code/`):

```env
GEMINI_API_KEY=
OPENAI_API_KEY=
LOG_LEVEL=INFO
DATA_DIR=../data
CHROMA_PERSIST_DIR=.chromadb
```

### Key environment variables
- `GEMINI_API_KEY`: preferred live provider key.
- `OPENAI_API_KEY`: optional fallback key.
- `LLM_ENABLE_CROSS_PROVIDER_FALLBACK=1`: enable Gemini -> OpenAI fallback chain.
- `LLM_ENABLE_MODEL_FALLBACKS=1`: allow trying additional Gemini model variants.
- `LLM_ENABLE_LIVE_API=1`: required in pytest mode to allow live API calls.

Notes:
- If provider quotas are exhausted, the agent falls back to local deterministic behavior.
- Keep `.env` out of source control.

## Quickstart (Local)

### Windows (PowerShell)
```powershell
cd code
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
copy code\.env.example .env
```

### macOS/Linux
```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
cp code/.env.example .env
```

Then populate `.env` with API keys if you want live LLM calls.

## Usage
Run from repository root using the project virtualenv Python.

### Full run
```powershell
.venv\Scripts\python.exe code\main.py
```

### Sample run (with internal scoring)
```powershell
.venv\Scripts\python.exe code\main.py --sample
```

### Dry run (first 5 tickets)
```powershell
.venv\Scripts\python.exe code\main.py --dry-run
```

### Custom input/output
```powershell
.venv\Scripts\python.exe code\main.py --input support_tickets\support_tickets.csv --output support_tickets\output.csv
```

## Docker Usage
Image name used below: `support-triage-agent`.

### Build
```powershell
docker build -t support-triage-agent code
```

### Run (full)
```powershell
docker run --rm --env-file .env -v "${PWD}/data:/data:ro" -v "${PWD}/support_tickets:/support_tickets" support-triage-agent python main.py
```

### Run (dry-run)
```powershell
docker run --rm --env-file .env -v "${PWD}/data:/data:ro" -v "${PWD}/support_tickets:/support_tickets" support-triage-agent python main.py --dry-run
```

### Compose
```powershell
docker compose up --build
```

## Architecture
High-level pipeline:
1. Ingestion: loads and parses local support corpus files.
2. Chunking: generates stable chunks with metadata.
3. Retrieval: hybrid dense+sparse retrieval, reranking, company-aware filtering.
4. Security: PII/prompt-injection detection and escalation triggers.
5. Agent: context assembly + structured output generation.
6. Sanitization: blocks unsafe/non-grounded responses.
7. Emission: writes `output.csv` and logs transcript/tool traces.

Key modules:
- `ingestion/`
- `retrieval/`
- `security/`
- `agent/`
- `pipeline.py`

## Quality Gates and Testing

### Full suite
```powershell
.venv\Scripts\python.exe -m pytest code\tests -q
```

### Phase slices
```powershell
.venv\Scripts\python.exe -m pytest code\tests\test_phase4_agent.py -q
.venv\Scripts\python.exe -m pytest code\tests\test_phase5_pipeline.py -q
.venv\Scripts\python.exe -m pytest code\tests\test_phase6_final.py -q
```

### Scoring (sample)
```powershell
.venv\Scripts\python.exe code\scorer.py --sample support_tickets\sample_support_tickets.csv --predictions support_tickets\output.csv
```

## Output Contract
`support_tickets/output.csv` fields:
- `ticket_id`
- `status` (`answered` | `escalated`)
- `product_area`
- `response`
 - `resolution` (short resolution/routing note)
- `justification`
- `request_type` (`product_issue` | `feature_request` | `bug` | `invalid`)

Constraints:
- No empty required fields.
- No `reasoning_trace` in CSV output.
- Status values must remain valid.
 - `resolution` should be present (can be short text like `see response` or `escalated:security`).

## Operational Notes
- Transcript logging is append-only and shared across agent sessions.
- LLM calls include quota mitigation: request pacing, short-output caps, cooldown-aware retries.
- If quotas are exhausted, pipeline continues with local deterministic fallback.

## Troubleshooting

### `docker: invalid env file ... contains whitespaces`
Ensure `.env` lines are strict `KEY=value` with no trailing spaces in key names.

### `Data directory does not exist: /data`
Use Docker mounts exactly as documented (`-v "${PWD}/data:/data:ro"`).

### `Input CSV not found: /support_tickets/support_tickets.csv`
Mount `support_tickets` to `/support_tickets` in container.

### `API key not valid` / `RESOURCE_EXHAUSTED`
- Verify keys in `.env`.
- Confirm provider quota/billing.
- If quota is exhausted, local fallback path still produces outputs.

### Slow first run
Initial corpus indexing can take longer; subsequent runs are faster due to persisted local index.
# Final Submission Checklist — Multi-domain Support Triage

Purpose: pre-submit gate to verify the project is complete, reproducible, and ready for evaluation.

## Quick Pass

- [ ] Include these files in the submission bundle
  - `code/` (all source code and `code/README.md`)
  - `support_tickets/` (CSV inputs and `output.csv`) 
  - `FINAL_SUBMISSION_CHECKLIST.md` (this file)
  - `.env.example` (no real keys)
  - `Dockerfile` and `docker-compose.yml` (if used)
  - `README.md` (repo root) and `code/README.md`

- [ ] Ensure no secrets are committed (`.env` must be gitignored)
- [ ] Ensure `hackerrank_orchestrate/log.txt` exists and includes recent session entries

## Environment & Setup

1. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code/requirements.txt
```

2. Prepare environment variables
- Copy `.env.example` -> `.env` and populate provider keys (e.g. `GEMINI_API_KEY` or `OPENAI_API_KEY`).
- Do NOT commit `.env`.

## Tests & Verification (must pass before submitting)

- [ ] Run unit/integration tests

```powershell
.\.venv\Scripts\python.exe -m pytest code/tests -q
```

- [ ] Confirm all tests pass. Record: `pytest` summary (example: `92 passed`).

## Run the agent (local) — quick checks

1. Dry run (no provider calls)

```powershell
python code/main.py --dry-run
```

2. Sample run (small sample set)

```powershell
python code/main.py --sample
python code/scorer.py --sample
```

3. Full run (produce `output.csv`)

```powershell
python code/main.py
```

4. Validate output
- Confirm `output.csv` row count equals `support_tickets/support_tickets.csv` input rows.
- Confirm no null required fields and schema contains expected columns (status, resolution, request_type, etc.).
- Run scorer if desired: `python code/scorer.py --input output.csv`.

## Docker (optional reproducible run)

- Build image

```powershell
docker build -t support-triage-agent ./code
```

- Run container (mount repo and provide `.env`)

```powershell
docker run --rm --env-file .env -v %CD%/data:/data:ro -v %CD%/support_tickets:/support_tickets support-triage-agent python main.py --sample
```

Note: builds may take time (dependency installs). Watch logs for provider 429/RESOURCE_EXHAUSTED messages — these are provider quota limits, not code failures.

## Logs & Diagnostics to include

- `hackerrank_orchestrate/log.txt` (home directory) — include last session entries.
- `code/agent/llm_client.py` diagnostic logs (if present in runtime output).
- `output.csv` (final run) and any scorer reports.

## Known Caveats & Recommendations

- Provider quotas: live LLM providers (Gemini / OpenAI) may return `RESOURCE_EXHAUSTED` or 429. Our client implements queueing, backoff, and cooldowns; if you need higher throughput, request quota increase from the provider or use a paid plan.
- Do not include real API keys in the repository. Use `.env.example` for submission and provide key instructions separately to evaluators if required.

## Submission packaging (recommended)

- Create a zip containing the required submission files:

PowerShell example:
```powershell
Compress-Archive -Path code, support_tickets, FINAL_SUBMISSION_CHECKLIST.md, code/README.md, README.md, .env.example -DestinationPath submission.zip
```

Or on Unix:
```bash
zip -r submission.zip code support_tickets FINAL_SUBMISSION_CHECKLIST.md code/README.md README.md .env.example
```

## Final sign-off (check all before uploading)

- [ ] `pytest` green (all tests passed)
- [ ] `output.csv` validated (rows, schema, nulls)
- [ ] No secrets committed (`git status` clean of `.env`)
- [ ] Docker image builds (if included)
- [ ] Log file `hackerrank_orchestrate/log.txt` included
- [ ] README files up to date (`README.md`, `code/README.md`)

- [x] README files up to date (`README.md`, `code/README.md`)

- [x] `pytest` green (all tests passed)
- [ ] `output.csv` validated (rows, schema, nulls)
- [x] No secrets committed (`git status` clean of `.env`)
- [x] Docker image builds (if included)
- [x] Log file `hackerrank_orchestrate/log.txt` included
- [ ] README files up to date (`README.md`, `code/README.md`)

- [x] `pytest` green (all tests passed)
- [x] `output.csv` validated (rows, schema, nulls)
- [x] No secrets committed (`git status` clean of `.env`)
- [x] Docker image builds (if included)
- [x] Log file `hackerrank_orchestrate/log.txt` included
- [ ] README files up to date (`README.md`, `code/README.md`)

If all boxes are checked, you're ready to submit.

---
Generated: 2026-05-02

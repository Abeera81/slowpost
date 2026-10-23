# Review Checklist

Do not mark work complete until the relevant checks pass.

## Basic Checks

- [ ] Diff is focused on the requested task.
- [ ] No unrelated files were rewritten.
- [ ] No secrets, tokens, private logs, or real account data were exposed. `git status` shows nothing from `data/`, `.venv/` or `.env`.
- [ ] Protected actions, if taken, were covered by the user's authorization as defined in AGENTS.md.
- [ ] `pytest` and `ruff check .` passed, or failures are explained.
- [ ] UI changes were checked in Chrome on the laptop (`streamlit run app.py`).

## Data and Model Integrity

- [ ] Backtest split is by `post_datetime`; no test row is older than any training row.
- [ ] Label thresholds come from training rows only.
- [ ] Only pre-posting features are used (list in AGENTS.md).
- [ ] Draft features are computed by code, not by Gemma.
- [ ] Every number shown in the UI or article comes from a real run file (`backtest.json`, run log). Nothing invented or rounded up.

## Security

- [ ] New dependencies are only the ones in the Tech Design, or the user approved them. `pip-audit` if time allows.
- [ ] CSV input is validated at load (required columns, missing or zero `reach`, too few rows).
- [ ] Not applicable: auth-protected routes (no accounts).
- [ ] Not applicable: rate limiting (local, single user, no public endpoint).

## AI Checks

- [ ] Model-visible data matches the AI section of AGENTS.md.
- [ ] The brief and captions are treated as untrusted data; the "ignore your rules" brief still gives 7 schema-valid drafts.
- [ ] Gemma output is schema-validated, retried once, and shown as "could not draft" after a second failure.
- [ ] Ollama-not-running and TabPFN-weights-missing give clear messages.
- [ ] No cloud AI calls: `tabpfn-client` is not installed or imported; the app works with Wi-Fi off after models are downloaded.
- [ ] Logs don't contain secrets.
- [ ] Not applicable: provider retention settings (no cloud provider). Builder export review (no builder used).

## Final Evidence

The final response should include:

- Files changed
- Commands run
- Test/build/browser results
- AI eval case results, if applicable
- Remaining risks

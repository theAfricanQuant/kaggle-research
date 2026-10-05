# Repository guidance

## Purpose and layout

This repository provides a reproducible research loop for tabular Kaggle, Zindi, and DrivenData competitions. The root `SKILL.md` is the installable agent entry point. The runnable project copied into each competition lives under `template/`; make runtime and workflow changes there, and update the root README when user-facing behavior changes.

The main flow is: inspect competition rules and data → choose task, metric, and CV → run hypotheses on frozen folds → retain every OOF/test prediction → gate adoption against the measured noise floor → build the final ensemble. Do not present a high CV score as trustworthy until the split scheme matches the competition.

## Competition correctness

- Use the competition's exact evaluation metric whenever it is supported. The current built-ins are listed in `template/main.py` and scored in `template/pipeline/validate.py`; keep metric names, score direction, scorer behavior, tuning, early stopping, noise-floor checks, submission selection, and docs consistent when adding one.
- If a competition uses a custom or unsupported metric, do not silently substitute ROC AUC, R², or another proxy. Add a correct scorer or make the limitation explicit before comparing models.
- Review competition rules for groups, time ordering, spatial structure, predefined splits, leakage, and submission schema. CSV heuristics are suggestions only. Configure `competition.json` or explicit CLI flags before the first run; folds are frozen and must not be reused after changing the split configuration.
- Never compute target-derived features from a row's own target. Target encodings and learned preprocessing must be fold-safe. Do not let validation rows determine preprocessing statistics used in their own fold.
- Preserve OOF comparability: every hypothesis in a run uses the same frozen folds and row order. Persist predictions and enough metadata to reproduce the experiment.
- A leaderboard submission is an external action. Only submit when the user explicitly authorizes `--submit`, and respect `--max-submissions`.

## Development workflow

- Use Python 3.11+ and `uv`; install dependencies with `uv sync` from `template/` when needed.
- Keep model integrations behind `template/pipeline/` and experiment routing in `template/main.py` / `template/worker.py`.
- When adding a hypothesis, register it in `worker.py` and the routing phases in `main.py`, return OOF and test predictions, and document the behavior.
- Keep the competition template self-contained: changes needed by generated competition projects must be made under `template/`.
- Run the focused validation tests with `uv run pytest` from `template/` when implementation changes affect scoring, folds, state, features, or submissions.
- Do not run a competition, download private data, or send leaderboard submissions as part of routine development.

## Competition brief

`competition.json` may specify `target_col`, `task`, `metric`, `cv_strategy`, `group_col`, `time_col`, and explanatory `notes`. Explicit CLI settings override corresponding brief settings. Keep the brief advisory: verify its contents against the competition's Evaluation page and data description.

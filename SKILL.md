---
name: kaggle-research
description: Run or improve reproducible binary-classification and regression workflows for Kaggle, Zindi, or DrivenData tabular competitions. Use for competition execution, CV design, tuning, experiment tracking, and OOF ensembling.
---

# kaggle-research

Run this skill only in a dedicated project directory. Its research loop is:
hypothesise → frozen-CV verification → noise-floor gate → persist → ensemble.

## Completion criteria

- Data, metric, task, and CV strategy are explicit in `state/log.json`.
- Every attempted hypothesis has comparable OOF predictions and reproducibility metadata.
- `submission_final.csv` exists when test data exists.
- Any leaderboard submission was explicitly authorized with `--submit` and stayed within its budget.
- The final report names the best CV result, ensemble score, validation risks, and artifact paths.

## Step-by-step execution

### 1. Copy template files into the current directory

The skill's base directory was shown when this skill was loaded. Copy everything from `template/` (not the skill's own `SKILL.md`/`README.md`) into the current working directory:

```bash
cp -r <skill-dir>/template/. .
```

Never run the loop from inside the skill's own directory — if the skill was installed via a symlink (common with `npx skills add`), writing state there would mutate the shared, shared-across-projects source.

### 2. Install dependencies

```bash
uv sync
```

### 3. Ensure Kaggle API token (Kaggle competitions only)

```bash
mkdir -p ~/.kaggle
```

If the token is missing, ask the user to create it at kaggle.com/account and place it there with mode `0600`. Check existence and permissions without printing its contents. Zindi and DrivenData use local data instead.

### 4. Configure and run

Read the README sections for the chosen platform and CV scheme. A safe local run writes submission files but makes no leaderboard submission:

```bash
uv run main.py --competition "<slug>" --iterations 50
```

Grouped and temporal competitions require an explicit split column:

```bash
uv run main.py --competition "<slug>" --group-col customer_id
uv run main.py --competition "<slug>" --time-col event_time
```

For Zindi/DrivenData, add `--data-path <folder>` and upload the generated CSV manually.
Pass `--submit --max-submissions N` only when the user explicitly authorizes Kaggle submissions.


### 5. Monitor and report

Check `state/log.json` after each experiment. Report:

- Best CV score and experiment.
- Accepted feature transforms and ensemble weights.
- Noise floor, adversarial-validation warning, and CV scheme.
- Final submission and experiment artifact paths.
- Leaderboard alignment only when authorized submissions exist.

## Routing

Each registered hypothesis runs at most once on the same frozen folds.
Feature transformations compound only when their gain exceeds the measured noise floor.
Optuna tuning follows diverse defaults; every result remains available to the final OOF hill-climbing ensemble.
When registered work is exhausted, stop rather than repeat identical experiments.


## File layout for reference

| File | Role |
|---|---|
| `template/main.py` | Orchestrator — loop, routing, CV gating, submission |
| `template/hardware.py` | GPU/RAM/core detection |
| `template/worker.py` | Hypothesis dispatcher |
| `template/pipeline/train.py` | Model trainers (default + tuned + depth-1) |
| `template/pipeline/tuner.py` | Optuna 5-stage stepwise tuning |
| `template/pipeline/validate.py` | CV scoring, task detection |
| `template/pipeline/features.py` | Feature engineering |
| `template/pipeline/ensemble.py` | Averaging + hill-climbing helpers |
| `template/pipeline/download.py` | Kaggle data download |
| `template/pipeline/submit.py` | Submission + score polling |
| `template/state/log.py` | Experiment logger |
| `template/state/run.py` | Run/data fingerprints and resume compatibility |
| `template/state/experiments.py` | Atomic OOF/test prediction artifacts and metadata |
| `template/tests/` | Regression tests for CV, state, ensembling, and submissions |

Full methodology (CV design, target encoding, ensembling strategy, submission policy) is documented in `README.md` — read it before making changes to the loop.

## Extending

To add a hypothesis, implement its handler in `worker.py`, register it in `KNOWN_HYPOTHESES`, add it to one routing phase in `main.py`, and add a regression test proving its OOF/test-prediction contract.

# kaggle-research

Autonomous Kaggle competition agent using the autoresearch pattern: **hypothesise → implement → CV verify → keep/revert → submit**. Designed for opencode/Claude Code agents.

## When to use

User says any of: "run a Kaggle competition", "compete for me", "autoresearch this competition", "iterate on my model", "find the best ensemble", "try different approaches in parallel".

## How it works

1. **Download** — `kagglehub` fetches competition data, detects hardware (GPU/RAM/cores)
2. **Baseline** — stratified 5-fold + LightGBM defaults, logs initial CV
3. **Hypothesis loop** — `main.py` spawns parallel worker agents, each tests one hypothesis
4. **Verify** — every change is CV-scored; only improvements are kept; worse changes are reverted and logged as negatives
5. **Submit** — every N iterations, the best model submits to Kaggle; CV vs LB correlation is tracked
6. **Final** — top-k models by CV are ensembled; 2 final submissions are selected

## Decision tree (next hypothesis router)

```
if no baseline yet:
  → stratified 5-fold + LightGBM defaults

elif CV stable but below top 20%:
  → feature engineering (high-cardinality → target encoding, dates → decompositions)

elif single model plateaued:
  → try a second model type → averaging → blending → stacking

elif CV != LB (correlation < 0.5):
  → adversarial validation → fix split strategy

elif <3 days remaining:
  → stop iterating → ensemble top-k by CV → submit best 2
```

## Task detection

The orchestrator auto-detects whether the competition is **classification** or **regression**:

- **Classification** (integer target, ≤20 unique values): ROC-AUC scoring, `StratifiedKFold`, `predict_proba` outputs
- **Regression** (float target, >10 unique values): R² scoring, `KFold`, direct `predict` outputs

Override with `--task classification` or `--task regression`.

## Hardware awareness

The orchestrator auto-detects GPU/VRAM/RAM and adjusts parallelism, model family, and batch sizes. See `hardware.py`.

## File layout

| File | Role |
|---|---|---|
| `pyproject.toml` | uv project config with dependencies |
| `.python-version` | Python 3.12 |
| `SKILL.md` | This file |
| `main.py` | Autoresearch orchestrator |
| `hardware.py` | GPU/RAM detection |
| `worker.py` | Parallel hypothesis runner |
| `pipeline/download.py` | kagglehub data fetcher |
| `pipeline/validate.py` | CV scoring |
| `pipeline/train.py` | Model training |
| `pipeline/features.py` | Feature engineering |
| `pipeline/ensemble.py` | Blending/stacking/averaging |
| `pipeline/submit.py` | Kaggle submission |
| `state/log.py` | Iteration logger |
| `kaggle_wrapper.ipynb` | Thin notebook for final Kaggle GPU runs |

## Setup

```bash
uv init --python 3.12
uv add kagglehub pandas numpy scikit-learn lightgbm xgboost catboost psutil
# API token: https://kaggle.com/account → Create API Token → ~/.kaggle/kaggle.json
```

## Run

```bash
uv run main.py --competition "tabular-playground-series-jan-2021" --iterations 50
```

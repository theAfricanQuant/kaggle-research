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
Phase 1 — defaults (iterations 1-10):
  iterations 1-4:   LightGBM baseline + feature engineering
  iterations 5-9:   try default XGBoost / CatBoost
  iteration 10:     Depth-1 XGBoost (GAM-like) ensemble

Phase 2 — Optuna tuning (from iteration 11, gated by CV score):
  low CV:      Optuna-tuned XGBoost (50 trials, TPE sampler)
  mid-low CV:  Optuna-tuned LightGBM
  mid CV:      Optuna-tuned CatBoost
  mid-high CV: Depth-1 XGBoost ensemble → average → blend → stack

At any point:
  if CV != LB (correlation < 0.5):
    → adversarial validation → fix split strategy
  if <3 days remaining:
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
| `bootstrap.sh` | Create a named competition folder from this template |
| `SKILL.md` | This file |
| `main.py` | Autoresearch orchestrator |
| `hardware.py` | GPU/RAM detection |
| `worker.py` | Parallel hypothesis runner |
| `pipeline/download.py` | kagglehub data fetcher |
| `pipeline/validate.py` | CV scoring, task detection |
| `pipeline/tuner.py` | Optuna 5-stage stepwise tuning (XGB/LGBM/Cat) |
| `pipeline/train.py` | Model training (default + tuned + depth-1) |
| `pipeline/features.py` | Feature engineering |
| `pipeline/ensemble.py` | Blending/stacking/averaging |
| `pipeline/submit.py` | Kaggle submission |
| `state/log.py` | Iteration logger |
| `kaggle_wrapper.ipynb` | Thin notebook for final Kaggle GPU runs |

## Setup

**Step 1 — Clone the template (one-time)**

```bash
git clone https://github.com/theAfricanQuant/kaggle-research.git
cd kaggle-research
```

**Step 2 — Bootstrap a named project folder**

```bash
./bootstrap.sh my-competition
cd my-competition
```

**Step 3 — Install dependencies**

```bash
uv sync
```

**Step 4 — Set up the Kaggle API token (one-time)**

```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 5 — Run**

```bash
uv run main.py --competition "my-competition-slug" --iterations 50
```

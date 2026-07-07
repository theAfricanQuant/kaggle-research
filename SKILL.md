# kaggle-research

Autonomous Kaggle/Zindi competition agent. Uses the **autoresearch pattern**: hypothesise → implement → CV verify → keep/revert → submit.

## When the user says

"Run competition X", "compete for me on Kaggle", "autoresearch this competition", "iterate on my model", "find the best ensemble".

## Your job (the agent)

Run experiments autonomously. The goal: find the best model for the competition by iterating through hypotheses, keeping only what improves CV score.

## Step-by-step execution

### 0. Setup

If this is the first run:

```bash
# Ensure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd kaggle-research
uv sync

# Ensure Kaggle API token exists
mkdir -p ~/.kaggle
# Ask user to download from kaggle.com/account → Create API Token
# and place at ~/.kaggle/kaggle.json
```

### 1. Scaffold a project folder

Create a folder named after the competition:

```bash
uv run main.py --competition "<slug>" --name <slug> --iterations 50
cd <slug>
uv sync
```

Ask the user to set their API token if not already done. Then proceed.

### 2. Run the autoresearch loop

```bash
uv run main.py --competition "<slug>" --iterations 50
```

### 3. Monitor progress

The agent should check `state/log.json` after each submission to track:
- **latest_cv** — is it trending up?
- **lb_score** — how does it compare to CV? (correlation check)
- **hypothesis** — what was tried, what worked

### 4. Intervene if needed

If CV-LB correlation < 0.3 (validation is broken), suggest adversarial validation or a different split strategy. If the user has ideas (e.g., "try this feature"), add them as hypothesis branches in `worker.py`.

## Decision tree (what to try next)

The orchestrator auto-routes, but here's the logic so you understand:

**Phase 1 — Defaults (iterations 1-10)**
Fast baselines, no tuning:
1-4: LightGBM defaults → feature engineering (target encoding, interactions)
5-9: Default XGBoost → default CatBoost
10: Depth-1 XGBoost + LightGBM ensemble

**Phase 2 — Optuna tuning (iterations 11+, gated by CV)**

Classification (ROC-AUC):
| Score | Try |
|---|---|
| <0.75 | Optuna-tuned XGBoost |
| 0.75-0.82 | Optuna-tuned LightGBM |
| 0.82-0.85 | Optuna-tuned CatBoost |
| 0.85-0.87 | Depth-1 ensemble |
| 0.87-0.88 | Average all 3 models |
| 0.88-0.90 | Blend (meta-model) |
| ≥0.90 | Stack ensemble |

Regression (R²):
| Score | Try |
|---|---|
| <0.55 | Optuna-tuned XGBoost |
| 0.55-0.65 | Optuna-tuned LightGBM |
| 0.65-0.70 | Optuna-tuned CatBoost |
| 0.70-0.75 | Depth-1 ensemble |
| 0.75-0.78 | Average all 3 |
| 0.78-0.82 | Blend (meta-model) |
| ≥0.82 | Stack |

## Architecture (so you can extend)

| File | Role |
|---|---|
| `main.py` | Orchestrator — loop, routing, CV gating, submission |
| `hardware.py` | GPU/RAM/core detection |
| `worker.py` | Hypothesis dispatcher (maps names to functions) |
| `pipeline/train.py` | All model trainers (default + tuned + depth-1) |
| `pipeline/tuner.py` | Optuna 5-stage stepwise tuning |
| `pipeline/validate.py` | CV scoring, task detection |
| `pipeline/features.py` | Feature engineering |
| `pipeline/ensemble.py` | Averaging helpers |
| `pipeline/download.py` | Kaggle data download |
| `pipeline/submit.py` | Submission + score polling |
| `state/log.py` | Experiment logger |

To add a new hypothesis: add a function in `worker.py` and map it in `_run_hypothesis`. The router in `main.py` auto-picks the next one based on CV.

## Files the agent should NOT modify

`pipeline/download.py`, `pipeline/submit.py`, `hardware.py`, `state/log.py` — these have no tuning value. Focus on `main.py` routing, `worker.py` dispatch, `pipeline/train.py` models, and `pipeline/features.py` engineering.

# kaggle-research

Autonomous Kaggle/Zindi competition agent. Designed to run inside AI coding agent harnesses (opencode, Claude Code, pi.dev, Codex).

The loop: hypothesise → implement → CV verify → keep/revert → submit.

## When the user says

"Run competition X", "compete for me on Kaggle", "iterate on my model", "find the best ensemble", "try different approaches in parallel", "use the kaggle-research skill".

## What you need to do

The user is in an empty project folder. Your job is to:

1. Copy this skill's template files into the current working directory (the skill's base directory is listed at the top of this file — copy everything from there)
2. Install dependencies
3. Ensure the Kaggle API token is set up
4. Scaffold a named project folder
5. Run the competition loop
6. Monitor progress and report back

## Step-by-step execution

### 1. Copy template files into the current directory

The skill's base directory was shown when this skill was loaded. Copy everything from there (except `.git/`, `__pycache__/`, `bootstrap.sh`) into the current working directory. Use `cp -r`, `rsync`, or `shutil.copytree` — whichever you prefer.

### 2. Install dependencies

```bash
uv sync
```

### 3. Ensure Kaggle API token

```bash
mkdir -p ~/.kaggle
```

If `~/.kaggle/kaggle.json` doesn't exist, ask the user to download it from kaggle.com/account → Create API Token and place it there.

### 4. Scaffold a project folder

```bash
uv run main.py --competition "<slug>" --name <slug> --iterations 50
cd <slug>
```

### 5. Run the autoresearch loop

```bash
uv run main.py --competition "<slug>" --iterations 50
```

### 6. Monitor

Check `state/log.json` after each submission block. Report to the user:
- Current best CV score
- What hypothesis worked best
- CV vs leaderboard alignment
- When iterations are done, the final score

## Decision tree

The orchestrator auto-routes, but here's the logic so you understand:

**Phase 1 — Defaults (iterations 1-10)**
Fast baselines, no tuning.
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
| 0.85-0.87 | TFDF (TensorFlow Decision Forests) |
| 0.87-0.88 | Depth-1 ensemble |
| 0.88-0.89 | Average all 3 models |
| 0.88-0.90 | Blend (meta-model) |
| ≥0.90 | Stack ensemble |

Regression (R²):
| Score | Try |
|---|---|
| <0.55 | Optuna-tuned XGBoost |
| 0.55-0.65 | Optuna-tuned LightGBM |
| 0.65-0.70 | Optuna-tuned CatBoost |
| 0.70-0.75 | TFDF (TensorFlow Decision Forests) |
| 0.75-0.78 | Depth-1 ensemble |
| 0.78-0.80 | Average all 3 |
| 0.78-0.82 | Blend (meta-model) |
| ≥0.82 | Stack |

## File layout for reference

| File | Role |
|---|---|
| `main.py` | Orchestrator — loop, routing, CV gating, submission |
| `hardware.py` | GPU/RAM/core detection |
| `worker.py` | Hypothesis dispatcher |
| `pipeline/train.py` | Model trainers (default + tuned + depth-1) |
| `pipeline/tuner.py` | Optuna 5-stage stepwise tuning |
| `pipeline/validate.py` | CV scoring, task detection |
| `pipeline/features.py` | Feature engineering |
| `pipeline/ensemble.py` | Averaging helpers |
| `pipeline/download.py` | Kaggle data download |
| `pipeline/submit.py` | Submission + score polling |
| `state/log.py` | Experiment logger |

## Extending

To add a new hypothesis: add a function in `worker.py` and map it in `_run_hypothesis`. The router in `main.py` auto-picks the next one based on CV.

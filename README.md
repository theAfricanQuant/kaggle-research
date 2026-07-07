# kaggle-research

An autonomous Kaggle competition agent that implements the **autoresearch** pattern: hypothesise → implement → CV verify → keep/revert → submit.

Supports both **classification** and **regression** tasks — auto-detected from the target column.

Designed for AI coding agents (Claude Code, Codex, Cursor) to run competitions end-to-end without manual iteration.

## How it works

```
main.py orchestrates the loop:
  → downloads competition data
  → detects hardware (GPU / RAM / cores)
  → spawns parallel worker agents
  → each worker tests one hypothesis (model type, feature transform, ensemble)
  → CV score gates every change: improvements kept, regressions reverted
  → every 5 iterations: submits best model to Kaggle
  → tracks CV vs leaderboard correlation
  → final push: ensemble top-k models
```

The decision tree automatically routes to the next hypothesis based on current CV score and iteration count:

### Classification (ROC-AUC)

| CV Score | Next Hypothesis |
|---|---|
| No baseline yet | Stratified 5-fold + LightGBM defaults |
| <0.70 | Feature engineering (target encoding) |
| <0.80 | Try XGBoost with tuning |
| <0.85 | Try CatBoost with tuning |
| <0.88 | Average LGBM + XGB + CatBoost |
| <0.90 | Blend with logistic regression meta-model |
| ≥0.90 | Stack ensemble of all top models |

### Regression (R²)

| CV Score | Next Hypothesis |
|---|---|
| No baseline yet | 5-fold + LightGBM defaults |
| <0.50 | Feature engineering (target encoding) |
| <0.65 | Try XGBoost with tuning |
| <0.75 | Try CatBoost with tuning |
| <0.80 | Average LGBM + XGB + CatBoost |
| <0.85 | Blend with ridge regression meta-model |
| ≥0.85 | Stack ensemble of all top models |

Hardware awareness auto-detects GPU/VRAM/RAM and adjusts parallelism, model depth, and batch sizes accordingly.

## Quick start

```bash
# 1. Install dependencies
pip install kagglehub pandas numpy scikit-learn lightgbm xgboost catboost torch psutil

# 2. Set up Kaggle API token
# Go to kaggle.com/account → Create API Token → save as ~/.kaggle/kaggle.json

# 3. Run
python main.py --competition "tabular-playground-series-jan-2021" --iterations 50
```

## Usage

```bash
python main.py \
  --competition "<competition-name>" \
  --iterations 50 \
  --submission-interval 5 \
  --final-days 3
```

| Flag | Default | Description |
|---|---|---|
| `--competition` | required | Kaggle competition slug (e.g., `tabular-playground-series-jan-2021`) |
| `--iterations` | 50 | Total hypotheses to test |
| `--submission-interval` | 5 | How often to submit to the leaderboard (skipped for first 10 iterations) |
| `--final-days` | 3 | When <3 days remain in competition, submit every iteration |
| `--task` | `auto` | Force task: `classification`, `regression`, or `auto`-detect |

## File layout

```
kaggle-research/
├── SKILL.md              → Agent entry point (triggers, workflow, decision tree)
├── main.py               → Orchestrator: the autoresearch loop
├── hardware.py           → GPU / RAM / core detection
├── worker.py             → Parallel hypothesis dispatcher
├── pipeline/
│   ├── download.py       → kagglehub data fetcher
│   ├── validate.py       → CV scoring
│   ├── train.py          → LGBM / XGBoost / CatBoost trainers
│   ├── features.py       → Target encoding, interaction features
│   ├── ensemble.py       → Averaging, blending, stacking
│   └── submit.py         → Submission + score polling
├── state/
│   └── log.py            → Iteration history logger
└── kaggle_wrapper.ipynb  → Thin notebook for final runs on Kaggle GPUs
```

## Agent integration

If you're using an AI coding agent (Claude Code, Codex), load the skill:

```bash
# The agent reads SKILL.md and understands the full workflow
# Then just say:
#   "Run the kaggle-research skill on competition X"
```

## Final submission flow

For the final push (last 3 days of a competition):

```bash
python main.py --competition "<name>" --iterations 50 --final-days 3
```

Then to generate a submission on Kaggle's own GPUs:

1. Run `python main.py --export-final` to save trained models
2. Upload the saved models as a Kaggle Dataset
3. Run `kaggle_wrapper.ipynb` on Kaggle Notebooks (GPU/TPU enabled)
4. Submit the generated CSV

## Logging

All iterations are logged to `state/log.json`:

```json
{
  "competition": "tabular-playground-series-jan-2021",
  "latest_cv": 0.892,
  "iterations": [
    {
      "iteration": 1,
      "hypothesis": "stratified_5fold_lgbm_defaults",
      "cv_before": null,
      "cv_after": 0.723,
      "delta": 0.723,
      "lb_score": 0.719,
      "timestamp": "2025-12-15T10:30:00"
    }
  ]
}
```

## Requirements

- Python 3.9+
- Kaggle API token (`~/.kaggle/kaggle.json`)
- Internet connection for data download and submission
- Optional: CUDA-capable GPU for deep learning branches

## Related

This skill is based on *The Kaggle Book* 2nd Ed. (Massaron, Tunguz, Banachewicz, Packt 2025). The `kaggle/` directory in this repo contains the companion teaching workspace with interactive lessons.

# kaggle-research

An autonomous agent that runs Kaggle (and Zindi, DrivenData, etc.) competitions for you — automatically exploring models, tuning hyperparameters, ensembling, and submitting.

Think of it as a **robust research assistant** that never forgets what it tried, never repeats a dead end, and always trusts cross-validation over leaderboard noise.

It implements the **autoresearch pattern**: hypothesise → implement → CV verify → keep/revert → submit. Originally described by [Andrej Karpathy](https://github.com/karpathy), now applied to data science competitions.

Supports **classification** and **regression** tasks — auto-detected from your data's target column. Designed to run anywhere: your laptop, a cloud VM, or even an AI coding agent (Claude Code, Codex).

---

## Table of Contents

- [Who is this for](#who-is-this-for)
- [What it does, step by step](#what-it-does-step-by-step)
- [The decision tree (how it chooses what to try next)](#the-decision-tree-how-it-chooses-what-to-try-next)
- [Prerequisites](#prerequisites)
- [Quick start: your first competition](#quick-start-your-first-competition)
- [Complete setup guide](#complete-setup-guide)
- [Running the agent](#running-the-agent)
- [Understanding the output](#understanding-the-output)
- [Example: walking through a real run](#example-walking-through-a-real-run)
- [Using with Zindi.africa](#using-with-zindiafrica)
- [Using with AI coding agents](#using-with-ai-coding-agents)
- [The final push: Kaggle Notebook submission](#the-final-push-kaggle-notebook-submission)
- [File-by-file breakdown](#file-by-file-breakdown)
- [FAQ](#faq)

---

## Who is this for

- **You're new to Kaggle** — the agent handles the full pipeline, so you can learn by watching what it tries and what works.
- **You're experienced but busy** — let the agent iterate through 50+ experiments while you focus on feature engineering or strategy.
- **You use Zindi.africa** — same workflow; just change the competition slug.
- **You use AI coding agents** — load `SKILL.md` and tell the agent to "run this competition."

---

## What it does, step by step

When you run `main.py`, here is exactly what happens:

### Step 1: Download the data

The agent uses `kagglehub` to download the competition's train/test CSVs. It caches them in `~/.cache/kagglehub/` so subsequent runs are instant.

### Step 2: Detect your hardware

It checks:
- **GPU** — does `torch.cuda.is_available()` return `True`? Which GPU? How much VRAM?
- **RAM** — how much system memory is available?
- **CPU cores** — how many parallel workers can it safely spawn?

This determines how many trees it builds, whether to enable GPU training (`task_type="GPU"` in CatBoost), and how many parallel hypothesis workers to run.

### Step 3: Detect the task type

It reads the training CSV, looks at the target column:
- If the target is integer with ≤20 unique values → **classification** (uses ROC-AUC, StratifiedKFold, predict_proba)
- If the target is float with >10 unique values → **regression** (uses R², KFold, direct predict)

You can override this with `--task classification` or `--task regression`.

### Step 4: Run the hypothesis loop

For each iteration (1 to N, default 50):

1. **Route** — the decision tree picks one hypothesis (e.g., "try Optuna-tuned XGBoost")
2. **Execute** — a worker trains the model using 5-fold cross-validation, storing out-of-fold predictions
3. **Score** — it computes the CV score (ROC-AUC for classification, R² for regression)
4. **Gate** — if this score is better than the previous best, keep the model; otherwise revert
5. **Log** — every hypothesis, score, and the delta are written to `state/log.json`
6. **Submit** — every N iterations (default 5, starting at iteration 10), it submits the best model to the real leaderboard and records the public LB score vs your CV score
7. **Track alignment** — it computes the correlation between your CV scores and leaderboard scores. If the correlation drops below 0.3, it warns you — this means your validation strategy is broken and the leaderboard is misleading you.

### Step 5: Final ensemble

When iterations run out, the agent has logged the best models by CV score. You can ensemble the top-k for a final push.

---

## The decision tree (how it chooses what to try next)

The agent has **two phases**. Phase 1 establishes baselines quickly with default parameters. Phase 2 brings out Optuna Bayesian optimisation once it understands the data.

### Phase 1 — Defaults (iterations 1-10)

These are fast, no-tuning runs to get a quick sense of the data:

| Iteration | What it tries | Why |
|---|---|---|
| 1 | LightGBM defaults (200-500 trees, 0.05 LR, 31 leaves) | Fastest tabular baseline |
| 2-3 | Feature engineering: target encoding on high-cardinality columns | Adds the most common competition trick |
| 4-5 | Feature engineering: pairwise interactions between top numeric features | Catches interactions LightGBM might miss |
| 6-7 | Default XGBoost (max_depth=6, 200-500 trees) | Different model family, different inductive bias |
| 8-9 | Default CatBoost (depth=6, 200-500 iterations) | Handles categoricals natively; often surprises |
| 10 | Depth-1 XGBoost + LightGBM ensemble | GAM-like additive model; cheap ensemble test |

### Phase 2 — Optuna Tuning (iterations 11+, gated by CV score)

Once baselines are done, the agent switches to **Optuna TPE** (Tree-structured Parzen Estimator) — the same Bayesian optimisation that Kaggle Grandmasters use. Each tuning run tests 50 parameter combinations with 5-fold CV and early stopping.

The decision is based on your current CV score:

**For classification (ROC-AUC):**

| CV Score | What it tries | Key params tuned |
|---|---|---|
| <0.75 | Optuna-tuned **XGBoost** | max_depth (3-12), min_child_weight, subsample, colsample, reg_alpha, reg_lambda, gamma, learning_rate |
| 0.75-0.82 | Optuna-tuned **LightGBM** | num_leaves (15-127), min_child_samples, subsample, colsample, reg_alpha, reg_lambda, learning_rate |
| 0.82-0.85 | Optuna-tuned **CatBoost** | depth (3-10), min_data_in_leaf, subsample, l2_leaf_reg, learning_rate |
| 0.85-0.87 | Depth-1 XGBoost + LightGBM avg | Cheap ensemble to test if additive structure helps |
| 0.87-0.88 | Average LGBM + XGB + CatBoost | Arithmetic mean of all 3; often beats any single one |
| 0.88-0.90 | Blending (LogisticRegression meta-model) | Trains a blender on holdout predictions from all 3 |
| ≥0.90 | Stacking (LogisticRegression on OOF features) | Full multi-layer stacking; highest potential, highest risk |

**For regression (R²):**

| CV Score | What it tries |
|---|---|
| <0.55 | Optuna-tuned XGBoost |
| 0.55-0.65 | Optuna-tuned LightGBM |
| 0.65-0.70 | Optuna-tuned CatBoost |
| 0.70-0.75 | Depth-1 XGBoost + LightGBM avg |
| 0.75-0.78 | Average LGBM + XGB + CatBoost |
| 0.78-0.82 | Blending (Ridge meta-model) |
| ≥0.82 | Stacking |

## Prerequisites

- **Python 3.9+** installed (3.12 recommended)
- **uv** — fast Python package manager. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Kaggle account** — sign up at [kaggle.com](https://kaggle.com)
- **Kaggle API token** — download from kaggle.com → Account → Create API Token → save to `~/.kaggle/kaggle.json`

Works on Linux, macOS, and Windows (WSL2). Works on both CPU-only machines and GPU-equipped ones (NVIDIA CUDA).

---

## Quick start: your first competition

Let's run the [Tabular Playground Series](https://kaggle.com/competitions/tabular-playground-series-jan-2021):

**Step 1 — Clone the template (one-time)**

```bash
git clone https://github.com/theAfricanQuant/kaggle-research.git
cd kaggle-research
```

**Step 2 — Create a named project folder**

```bash
./bootstrap.sh playground-jan-2021
cd playground-jan-2021
```

Or scaffold directly with `main.py` — no bootstrap.sh needed:

```bash
uv run main.py \
  --competition "tabular-playground-series-jan-2021" \
  --name playground-jan-2021 \
  --iterations 50
```

This creates the folder, copies all template files, updates `pyproject.toml`, inits git, and exits. Send it to a custom location:

```bash
uv run main.py \
  --competition "tabular-playground-series-jan-2021" \
  --name playground-jan-2021 \
  --out ~/Documents/07_DataScience/competition \
  --iterations 50
```

**Step 3 — Install dependencies**

```bash
uv sync
```

**Step 4 — Set up your Kaggle API token (one-time)**

```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Go to [kaggle.com/account](https://kaggle.com/account) → Create API Token to download `kaggle.json`.

**Step 5 — Run the agent**

```bash
uv run main.py --competition "tabular-playground-series-jan-2021" --iterations 50
```

**Step 1 — Clone the template (one-time)**

```bash
git clone https://github.com/theAfricanQuant/kaggle-research.git
cd kaggle-research
```

**Step 2 — Create a named project folder**

```bash
./bootstrap.sh my-competition
cd my-competition
```

Or scaffold directly:

```bash
uv run main.py --competition "<slug>" --name my-competition --iterations 50
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
uv run main.py --competition "<competition-slug>" --iterations 50
```

That's it. The agent will start iterating immediately. Next competition? Same steps:

```bash
cd ~/kaggle-research
./bootstrap.sh house-prices-advanced-regression-techniques
cd house-prices-advanced-regression-techniques
uv sync
uv run main.py --competition "house-prices-advanced-regression-techniques" --iterations 50
```

Each competition gets its own folder with its own `.git`, its own `state/log.json`, and its own `pyproject.toml` named after the competition. You'll see output like:

```
2026-07-07 10:30:00 | INFO | Hardware: GPU=True (NVIDIA RTX 4090) RAM=32GB Cores=16
2026-07-07 10:30:02 | INFO | Auto-detected task: classification
2026-07-07 10:30:02 | INFO | === Iteration 1/50 ===
2026-07-07 10:30:02 | INFO | Hypothesis: stratified_5fold_lgbm_defaults
2026-07-07 10:30:45 | INFO | 📋 CV: — → 0.7234 (Δ+0.7234)
2026-07-07 10:30:45 | INFO | === Iteration 2/50 ===
2026-07-07 10:30:45 | INFO | Hypothesis: feature_engineering_target_encoding
...
```

Let it run. Check the results with:

```bash
cat state/log.json | python -m json.tool
```

---

## Complete setup guide (for beginners)

### Installing Python

If you don't have Python 3.9+:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install python3 python3-pip -y
```

**macOS:**
```bash
brew install python@3.12
```

**Windows:** Download from [python.org](https://python.org) — ensure "Add Python to PATH" is checked.

### Installing uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal or run `source ~/.bashrc` (or `source ~/.zshrc`).

### Verifying the Kaggle API token

```bash
# Check that the token exists and is readable:
cat ~/.kaggle/kaggle.json | head -5
# You should see something like {"username":"yourname","key":"abc123..."}

# If it doesn't exist, re-download from kaggle.com/account
```

### Verifying GPU support (optional)

If your machine has an NVIDIA GPU:

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# If this errors, you're CPU-only — that's fine, the agent adapts.
```

---

## Running the agent

```bash
uv run main.py --competition "<competition-slug>" --iterations 50
```

### All command-line flags

| Flag | Default | What it does |
|---|---|---|---|
| `--competition` | (required) | Competition slug. Kaggle: `"tabular-playground-series-jan-2021"`. Zindi: `"zindi-competition-name"` |
| `--name` | none | Scaffold a new project folder with this name, copy all files, init git, then exit |
| `--out` | current dir | Parent directory for `--name` (e.g., `~/Documents/07_DataScience/competition`) |
| `--iterations` | 50 | Total hypotheses to test |
| `--submission-interval` | 5 | How often to submit to the leaderboard (starts at iteration 10) |
| `--metric` | `auto` | Optimisation metric. Classification: `roc_auc`, `logloss`, `accuracy`, `f1`. Regression: `rmse`, `mae`, `r2` |
| `--optuna-trials` | 50 | Hyperparameter trials per tuning session |
| `--task` | `auto` | Force task type: `classification`, `regression` |

### Example: tuning for RMSE on a regression competition

```bash
uv run main.py \
  --competition "house-prices-advanced-regression-techniques" \
  --iterations 40 \
  --metric rmse \
  --optuna-trials 80
```

### Example: quick run (10 iterations) to test things work

```bash
uv run main.py \
  --competition "tabular-playground-series-jan-2021" \
  --iterations 10 \
  --submission-interval 999  # never submit during testing
```

---

## Understanding the output

### Console log

Each iteration prints something like:

```
2026-07-07 10:35:00 | INFO | === Iteration 15/50 ===
2026-07-07 10:35:00 | INFO | Hypothesis: optuna_xgb
2026-07-07 10:35:00 | INFO |   Optuna tuning xgb (50 trials, metric=roc_auc)...
2026-07-07 10:38:30 | INFO |   Best xgb params: {'max_depth': 8, 'min_child_weight': 3,
  'subsample': 0.85, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.1,
  'gamma': 0.1, 'learning_rate': 0.023}
2026-07-07 10:39:00 | INFO | ✅ CV: 0.8102 → 0.8241 (Δ+0.0139)
```

Reading this:
- **Iteration 15** — the 15th hypothesis being tested
- **hypothesis: optuna_xgb** — the agent chose to run Optuna on XGBoost
- **50 trials** — it tested 50 parameter combinations internally
- **Best params** — the optimal params found (depth 8, LR 0.023, etc.)
- **✅ CV: 0.8102 → 0.8241** — the previous best was 0.8102; this model scored 0.8241. Improvement of 0.0139. Kept.

If the line says **❌ No improvement (Δ-0.0042)**, the model was worse. It's discarded. The previous best is preserved.

### State log (`state/log.json`)

After 50 iterations, this file contains the full history:

```json
{
  "competition": "tabular-playground-series-jan-2021",
  "task": "classification",
  "metric": "roc_auc",
  "latest_cv": 0.8921,
  "iterations": [
    {
      "iteration": 1,
      "hypothesis": "stratified_5fold_lgbm_defaults",
      "cv_before": null,
      "cv_after": 0.7234,
      "delta": 0.7234,
      "lb_score": null,
      "timestamp": "2026-07-07T10:30:00"
    },
    {
      "iteration": 10,
      "hypothesis": "depth1_xgb_ensemble",
      "cv_after": 0.8102,
      "lb_score": 0.8050,
      "delta": 0.0051,
      "timestamp": "2026-07-07T10:45:00"
    },
    {
      "iteration": 15,
      "hypothesis": "optuna_xgb",
      "cv_before": 0.8102,
      "cv_after": 0.8241,
      "delta": 0.0139,
      "lb_score": 0.8190,
      "timestamp": "2026-07-07T10:39:00"
    }
  ]
}
```

Key things to watch:
- **latest_cv** trend — is it going up? If it plateaus early, your data might be too noisy or your feature engineering needs work
- **lb_score vs cv_after** — if CV says 0.85 but LB says 0.72, your validation is broken (see CV-LB correlation warnings)
- **delta** — big jumps (+0.01+) are genuine breakthroughs; tiny jumps (<0.001) might be noise

### CV-LB alignment warnings

After 3+ submissions, the agent computes the Pearson correlation between your CV scores and leaderboard scores:

```
CV-LB correlation (last 5 submissions): 0.912  ← great, you can trust your CV
CV-LB correlation (last 5 submissions): 0.124  ← ⚠️ red flag
```

A correlation below 0.3 means your local validation doesn't match the leaderboard. Common causes:
- The test set has a different distribution than training (run adversarial validation)
- You're using random splits on time-series data (use TimeSeriesSplit)
- Data leakage in your training features

---

## Example: walking through a real run

Let's say you join the [House Prices competition](https://kaggle.com/competitions/house-prices-advanced-regression-techniques) — a regression problem.

```bash
uv run main.py --competition "house-prices-advanced-regression-techniques" --iterations 50
```

**Iterations 1-4:** LightGBM baseline. It detects regression (target is float with many unique values), uses KFold instead of StratifiedKFold, and scores with R². Let's say it gets R² = 0.78.

**Iterations 5-9:** Feature engineering. Target encoding and interaction features push R² to 0.82.

**Iteration 10:** Depth-1 XGBoost + LightGBM ensemble. R² stays at 0.82.

**Iteration 11 (Phase 2 starts):** CV is 0.82 which is ≥0.55 in the regression thresholds, so it skips XGBoost tuning and goes straight to Optuna-tuned LightGBM. After 50 trials, it finds better params: R² jumps to 0.85.

**Iteration 16:** Optuna-tuned CatBoost pushes to 0.86.

**Iteration 20:** Average LGBM + XGB + CatBoost → 0.87.

**Iteration 25:** Blending with Ridge → 0.88.

**Iteration 30+:** Stacking → 0.89.

At iteration 10, 15, 20, 25, etc., the agent submits to the real leaderboard. Your local CV shows 0.89, and the LB shows 0.88 — correlation is high, so you can trust the CV.

At the end, `state/log.json` shows the progression and you know your best model.

---

## Using with Zindi.africa

Zindi is Africa's data science competition platform. To use this agent with Zindi:

1. **Zindi doesn't have a direct Python API** like kagglehub. Instead, download the data manually:
   - Log in to [zindi.africa](https://zindi.africa)
   - Go to the competition page → Data tab → download Train.csv and Test.csv
   - Place them in a local folder

2. **Modify the download step** — either:
   - Write your own `pipeline/download.py` that reads from a local folder, or
   - Use the helper: create a folder with `train.csv` and `test.csv`, then set `data_path` in code

3. **Skip automatic submission** — Zindi requires manual CSV upload. Run the loop with `--submission-interval 999` to disable auto-submit, then manually upload `submission.csv` from the best iteration.

4. **Everything else works the same** — feature engineering, Optuna tuning, ensembling, CV scoring. The agent doesn't care where the data came from.

---

## Using with AI coding agents

If you use Claude Code, Codex, or Cursor:

1. Point the agent to the `SKILL.md` file. It contains the full workflow description in a format agents understand natively.
2. Say: "Load the kaggle-research skill and run it on competition X."
3. The agent reads the decision tree, understands the pipeline, and can execute the loop, interpret results, and iterate deeper on promising paths.

The `SKILL.md` and all Python code together let the agent act as a **Kaggle teammate** — one that never sleeps, never forgets what it tried, and always checks CV before declaring victory.

---

## The final push: Kaggle Notebook submission

When you're in the last days of a competition and want to use Kaggle's free GPUs:

1. Run the agent with more iterations to find the best ensemble:
   ```bash
   uv run main.py --competition "<name>" --iterations 80
   ```

2. The agent saves the best models in memory during the run. For a more permanent approach, you can extend the code to pickle models (see `kaggle_wrapper.ipynb`).

3. Upload the pickled models as a Kaggle Dataset:
   ```bash
   # From the Kaggle Notebook interface:
   # Add Data → Upload → select your .pkl files
   ```

4. Open `kaggle_wrapper.ipynb` in Kaggle Notebooks, point it to your uploaded dataset, and run it on a GPU/TPU kernel. It loads the models, predicts on the full test set, and generates `submission.csv`.

5. Download and submit the CSV.

---

## File-by-file breakdown

```
kaggle-research/
│
├── pyproject.toml           ← uv project config. Lists all Python dependencies.
│                              Run `uv sync` to install everything.
│
├── .python-version          ← Tells uv to use Python 3.12.
│
├── bootstrap.sh             ← Create a named competition folder from this template.
│                              Usage: `./bootstrap.sh my-competition-name`
│
├── SKILL.md                 ← Entry point for AI coding agents. Describes the full
│                              workflow, triggers, and decision tree in a format
│                              Claude Code / Codex understand natively.
│
├── README.md                ← This file.
│
├── main.py                  ← The orchestrator. The loop that ties everything together:
│                                1. Parses CLI arguments
│                                2. Detects hardware and task
│                                3. For each iteration: routes to next hypothesis,
│                                   dispatches workers, gates by CV, logs, submits
│                                4. Tracks CV-LB alignment
│
├── hardware.py              ← GPU / RAM / CPU core detection. Returns a dict
│                              that every model trainer uses to adjust depth,
│                              parallelism, and batch sizes.
│
├── worker.py                ← Hypothesis dispatcher. Maps hypothesis names to
│                              actual function calls. Each function trains a model,
│                              runs CV, and returns the score.
│
├── pipeline/
│   ├── download.py          ← Downloads competition data via kagglehub.
│                              Caches to ~/.cache/kagglehub.
│   │
│   ├── validate.py          ← CV scoring + task detection. Detects classification
│                              vs regression, picks the right splitter (StratifiedKFold
│                              vs KFold), and computes the right metric.
│   │
│   ├── tuner.py             ← Optuna hyperparameter tuning. Defines the search space
│                              for XGBoost, LightGBM, and CatBoost, runs TPE Bayesian
│                              optimisation, and returns the best params.
│   │
│   ├── train.py             ← All model trainers. Three variants per model family:
│                              - default (fast, fixed params)
│                              - tuned (accepts Optuna best_params)
│                              - depth-1 (max_depth=1 for GAM-like structure)
│   │
│   ├── features.py          ← Feature engineering: target encoding, pairwise
│                              interactions, categorical encoding.
│   │
│   ├── ensemble.py          ← Averaging, inverse-covariance weighting.
│                              Not called directly from the loop (blending/stacking
│                              are in worker.py).
│   │
│   └── submit.py            ← Kaggle submission via kagglehub. Polls for score
│                              after submit. Saves submission CSVs.
│
├── state/
│   └── log.py               ← JSON logger. Reads/writes state/log.json with
│                              full iteration history.
│
└── kaggle_wrapper.ipynb     ← Jupyter notebook for the final Kaggle GPU run.
                                Loads pickled models, generates submission CSV.
```

---

## FAQ

**How long does 50 iterations take?**
Depends on your data size and hardware. Small datasets (<10k rows, <100 columns): ~30-60 minutes. Medium datasets: ~2-4 hours. Large datasets: can take overnight. The first 10 iterations (defaults, no tuning) are much faster — tuning 50 Optuna trials per model is where the time goes.

**Can I stop and resume?**
Yes. The agent reads `state/log.json` at startup. If it exists, it continues from where it left off. Delete the file if you want a clean start.

**I don't have a GPU. Will this work?**
Yes. The hardware detector sets `hw["gpu"] = False`, which reduces tree counts and disables `task_type="GPU"`. All models train on CPU. It's slower but works fine.

**I don't have a Kaggle account. Can I still use this?**
You can use the pipeline components (training, tuning, CV, ensembling) on any CSV data. Just skip the submission step. The code works with any `train.csv`/`test.csv` structure.

**What if my competition uses a metric not listed?**
The `--metric` flag supports the most common ones (ROC-AUC, logloss, RMSE, MAE, R², accuracy, F1). If your competition uses something else (quadratic weighted kappa, MAP@K, etc.), you can add it to `pipeline/tuner.py` (the `optuna_metric` function) and it will work everywhere.

**Does it work with Zindi?**
Yes — see [Using with Zindi.africa](#using-with-zindiafrica). The main difference is you download data manually and upload submissions manually. The modelling loop is identical.

**Can I add my own hypothesis?**
Add a branch in `worker.py` (`_run_hypothesis`), add a model trainer in `pipeline/train.py`, and add the routing in `main.py` (`route_next_hypothesis`). Then the decision tree will call it automatically.

---

## Related

This tool is based on *The Kaggle Book* 2nd Ed. (Massaron, Tunguz, Banachewicz, Packt 2025) and *Effective XGBoost* (2nd Ed.), synthesised into an autonomous research agent using the Karpathy-style skill pattern.

The companion teaching workspace lives in the `kaggle/` subdirectory with interactive HTML lessons, a competition glossary, and reference materials.

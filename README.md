# kaggle-research

An autonomous competition agent that runs Kaggle, Zindi, and DrivenData competitions for you. Implements a **research loop**: hypothesise → implement → CV verify → keep/revert → submit — grounded in the practices Kaggle Grandmasters actually use (frozen CV folds, noise-floor gating, out-of-fold target encoding, hill-climbing ensembles).

Supports **binary classification** and **regression** — auto-detected from your data. (Multiclass isn't supported yet; the pipeline tells you explicitly rather than producing silently wrong predictions.)

## Quick start

### For AI agents (opencode, Claude Code, Codex, pi.dev)

**Step 1 — Install the skill (one-time)**

```bash
npx skills add theAfricanQuant/kaggle-research
```

This detects your installed agents (Claude Code, Codex, opencode, pi, and 60+ others) and writes the skill into each one's directory. If you're on WSL, pass `--copy` instead of the default symlink — symlinks across the Windows/WSL boundary can be unreliable.

Manual install, if you'd rather not use the installer:

```bash
git clone https://github.com/theAfricanQuant/kaggle-research.git
cp -r kaggle-research ~/.agents/skills/kaggle-research   # Codex, opencode, pi, Gemini CLI
cp -r kaggle-research ~/.claude/skills/kaggle-research    # Claude Code (reads its own directory only)
```

**Step 2 — Use it**

```bash
mkdir my-competition && cd my-competition
```

Then tell your agent:

> "Use the kaggle-research skill on competition house-prices-advanced-regression-techniques"

The agent reads `SKILL.md`, copies the `template/` files into your folder, installs deps, scaffolds a project, and runs the full research loop — all autonomously. Just ensure `~/.kaggle/kaggle.json` exists (Kaggle competitions only).

### Manual (without an agent)

```bash
git clone https://github.com/theAfricanQuant/kaggle-research.git
cd kaggle-research/template
uv sync
uv run main.py --name house-prices
cd house-prices
uv run main.py --competition "house-prices-advanced-regression-techniques" --iterations 50
```

---

## What it does, step by step

### Step 1: Download the data

For Kaggle competitions, `kagglehub` downloads the train/test CSVs and caches them in `~/.cache/kagglehub/`. For Zindi/DrivenData (no download API — see [Using with Zindi](#using-with-zindiafrica)), pass `--data-path <folder>` pointing at your manually downloaded `train.csv`/`test.csv`.

### Step 2: Detect your hardware

Checks GPU (`torch.cuda.is_available()`), RAM, and CPU cores. Determines tree counts and whether to enable GPU training (`task_type="GPU"` in CatBoost).

### Step 3: Detect the task type and freeze the CV folds

Reads the target column: integer/low-cardinality → **classification** (StratifiedKFold, ROC-AUC, predict_proba); float/high-cardinality → **regression** (KFold, R², direct predict). Override with `--task`.

The 5-fold split is generated **once** and written to `state/folds.json`. Every hypothesis for the rest of the run reuses those exact folds — this is what makes OOF predictions from different experiments comparable and stackable later. Changing folds mid-run would silently invalidate every ensembling step.

An **adversarial validation** check also runs at startup: it trains a classifier to distinguish train rows from test rows. AUC near 0.5 means your CV should track the leaderboard; AUC well above 0.5 means the train/test distributions differ, and the agent will warn you that local CV may not transfer.

### Step 4: Estimate the CV noise floor

Before the loop starts, the baseline model is trained a few times with different fold-shuffle seeds (`--noise-seeds`, default 3) to measure how much the CV score moves from randomness alone. Every subsequent hypothesis must beat this noise floor to be adopted as the new best — a naive "keep if CV improves at all" rule ratchets upward on noise, which is a well-documented failure mode in both human and LLM-agent Kaggle attempts.

### Step 5: Run the hypothesis loop

For each iteration:

1. **Route** — picks the next hypothesis that hasn't been tried yet (see [routing](#routing-untried-work-not-absolute-thresholds) below)
2. **Execute** — trains on the frozen folds, producing out-of-fold predictions (for CV scoring) and test-set predictions (for submission)
3. **Score** — computes the CV score on your chosen metric
4. **Gate** — keeps the new feature set / hyperparameters as the running baseline only if the improvement exceeds the noise floor; otherwise the experiment is logged but not adopted
5. **Persist** — every experiment's OOF and test predictions are saved to `state/experiments/`, win or lose — a "worse" model can still add value to the final ensemble through diversity
6. **Log** — hypothesis, CV score, and delta written to `state/log.json`
7. **Submit** — every N iterations (default 5, starting at iteration 10), submits the current best experiment's test predictions via the official `kaggle` CLI (installed as a project dependency; authenticates from the same `~/.kaggle/kaggle.json`). Submission headers and ids come from the competition's `sample_submission.csv`, so files aren't rejected over column names
8. **Track alignment** — after 5+ submissions, computes the Spearman rank correlation between CV and leaderboard scores; warns below 0.3

### Step 6: Final hill-climbing ensemble

Once all hypotheses are exhausted, the agent runs **Caruana-style hill climbing** over every persisted experiment: starting from the single best model, it greedily adds (with replacement) whichever library member most improves the blended OOF score, stopping when nothing helps. This routinely beats any single model and beats the old fixed average/blend/stack ladder, because it searches the full experiment history instead of a hand-picked subset — and because it never re-fits or re-scores on data it's judging, it doesn't leak. The blended test prediction is always written to `submission_final.csv` (using the competition's real submission headers), even on short runs that never hit the periodic-submission threshold.

---

## Routing: untried work, not absolute thresholds

The original design routed hypotheses off fixed CV-score bands (e.g. "AUC 0.82–0.85 → try CatBoost"). That doesn't generalize: an AUC of 0.75 is a winning score in some competitions and a broken baseline in others. The router now simply works through a fixed phase order and picks the first hypothesis not yet tried in this run:

**Phase 1 — fast baselines (no tuning):**
`lgbm_defaults → fe_target_encoding → fe_frequency → fe_interactions → xgb_defaults → catboost_defaults → depth1_xgb_ensemble`

Feature-engineering hypotheses that beat the noise floor become the new baseline feature set — later hypotheses build on the winning features instead of starting over from raw data each iteration.

**Phase 2 — Optuna tuning (TPE, 50 trials by default):**
`optuna_xgb → optuna_lgbm → optuna_catboost`

Each tuning run uses the same frozen folds and early-stops each fold's training — and, unlike the naive version, the tuned model's chosen number of trees (`mean_best_iteration` from the winning trial) is actually carried into the final fit, instead of silently falling back to a default tree count.

When every hypothesis has been tried, the loop ends early and moves to the final hill-climbing ensemble — it doesn't keep re-running the same exhausted hypotheses.

---

## Feature engineering

- **Target encoding** is out-of-fold (each fold's encoding is computed only from the other folds) and **smoothed toward the global mean** (m-estimate, m=300) so a category seen only a couple of times doesn't get treated as a confident predictor. Test-set encoding uses the full training set's statistics; unseen categories fall back to the global mean.
- **Categorical columns are kept as categoricals** and passed natively to CatBoost and LightGBM — they are not blanket-dropped or forced through unsmoothed raw target means, which is the classic silent regression from earlier tabular pipelines.
- **Frequency encoding** adds a simple count-based feature per categorical column.
- **Numeric interactions** are limited to pairwise products among the highest-variance numeric columns — GBDTs already model most interactions internally, so this targets ones a tree needs many splits to approximate rather than generating hundreds of low-value columns.

---

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Kaggle account** and **API token** (Kaggle competitions only) — kaggle.com/account → Create API Token → `~/.kaggle/kaggle.json`

Works on Linux, macOS, Windows (WSL2), CPU-only or NVIDIA GPU.

---

## Running the agent

```bash
uv run main.py --competition "<competition-slug>" --iterations 50
```

### All command-line flags

| Flag | Default | What it does |
|---|---|---|
| `--competition` | (required) | Competition slug/name, used for Kaggle download and submission messages |
| `--data-path` | none | Local folder with `train.csv`/`test.csv`. Required for Zindi/DrivenData; optional override for Kaggle |
| `--name` | none | Scaffold a new project folder with this name, copy all files, init git, then exit |
| `--out` | current dir | Parent directory for `--name` |
| `--iterations` | 50 | Maximum hypotheses to test (the loop stops early once all are tried) |
| `--submission-interval` | 5 | How often to submit to the leaderboard (starts at iteration 10) |
| `--metric` | `auto` | Optimisation metric. Classification: `roc_auc`, `logloss`, `accuracy`, `f1`. Regression: `rmse`, `mae`, `r2` |
| `--optuna-trials` | 50 | Hyperparameter trials per tuning session |
| `--task` | `auto` | Force task type: `classification`, `regression` |
| `--noise-seeds` | 3 | Seeds used to estimate the CV noise floor before the loop starts |

### Example: quick run to test things work

```bash
uv run main.py \
  --competition "tabular-playground-series-jan-2021" \
  --iterations 10 \
  --optuna-trials 10 \
  --submission-interval 999  # never submit during testing
```

---

## Understanding the output

### State log (`state/log.json`)

```json
{
  "competition": "tabular-playground-series-jan-2021",
  "task": "classification",
  "metric": "roc_auc",
  "latest_cv": 0.8921,
  "tried_hypotheses": ["lgbm_defaults", "fe_target_encoding", "..."],
  "final_ensemble_weights": {"optuna_lgbm_15": 2, "catboost_defaults_6": 1},
  "iterations": [
    {
      "iteration": 1,
      "hypothesis": "lgbm_defaults",
      "cv_before": null,
      "cv_after": 0.7234,
      "delta": 0.7234,
      "experiment_path": "state/experiments/lgbm_defaults_1.npz",
      "lb_score": null
    }
  ]
}
```

### CV-LB alignment

After 5+ submissions, the agent computes the Spearman rank correlation between your CV scores and leaderboard scores. A correlation below 0.3 means your local validation doesn't match the leaderboard — check for train/test distribution shift (the adversarial-validation AUC printed at startup), leakage in your features, or the wrong CV scheme for your data (grouped or time-ordered data needs `GroupKFold`/`TimeSeriesSplit`, which `get_splitter` in `pipeline/validate.py` doesn't auto-detect from the CSV alone).

### Experiment library (`state/experiments/*.npz`)

Every hypothesis's out-of-fold and test predictions, including the ones that didn't beat the running best — this is what the final hill-climbing ensemble searches over. `state/experiments.py` has the load/save helpers if you want to inspect or reuse them.

---

## Using with Zindi.africa

Zindi is Africa's data science competition platform, and has real differences from Kaggle worth knowing:

1. **No official API.** Download Train.csv/Test.csv manually from the competition's Data tab, put them in a folder, and run with `--data-path <folder>`.
2. **Submission is manual.** The agent still generates `submission.csv` from the current best experiment and `submission_final.csv` from the hill-climbed ensemble at the end of the run — upload these yourself. Auto-submission is skipped automatically when `--data-path` is set.
3. **Submission budgets are often capped for the whole competition**, not just per day — spend them on genuinely different pipelines, not on chasing the public leaderboard.
4. **Explicitly select your final 2 submissions.** If you don't, Zindi defaults to your best *public-LB* submissions — which is exactly the leaderboard-overfitting failure mode this agent's noise-floor gate and CV-trust design are meant to avoid. Pick your best-CV run and your most diverse strong ensemble.
5. **Top finishers must reproduce their code.** Frozen folds + the experiment log give you exactly what you need to reproduce any result on request.

---

## The final push: Kaggle Notebook submission

When you're in the last days of a competition and want Kaggle's free GPUs:

1. Run the agent with more iterations and a larger `--optuna-trials` to find the best ensemble.
2. `state/experiments/*.npz` has every model's OOF predictions; extend the code to pickle the fitted models if you want to reproduce them outside this run (see `kaggle_wrapper.ipynb`).
3. Upload the pickled models as a Kaggle Dataset, open `kaggle_wrapper.ipynb` on a GPU/TPU kernel, predict on the full test set, and generate `submission.csv`.
4. Download and submit.

---

## File-by-file breakdown

```
kaggle-research/
│
├── SKILL.md                 ← Entry point for AI coding agents (Claude Code, Codex,
│                              opencode, pi.dev). Spec-compliant frontmatter +
│                              step-by-step instructions.
│
├── README.md                ← This file.
│
├── TREE.md                  ← Directory layout reference.
│
└── template/                ← Everything an agent copies into your project folder.
    │
    ├── pyproject.toml       ← uv project config. Run `uv sync` to install everything.
    ├── .python-version      ← Python 3.12.
    ├── bootstrap.sh         ← Create a named competition folder from this template
    │                          without an agent: `./bootstrap.sh my-competition-name`
    │
    ├── main.py               ← The orchestrator: hardware/task detection, frozen-fold
    │                           setup, adversarial validation, noise-floor estimation,
    │                           the iteration loop, submission, and the final
    │                           hill-climbing ensemble.
    │
    ├── hardware.py           ← GPU / RAM / CPU core detection.
    │
    ├── worker.py             ← Hypothesis dispatcher. Every hypothesis function
    │                           returns OOF predictions + test predictions so they
    │                           can be persisted for the final ensemble.
    │
    ├── pipeline/
    │   ├── download.py       ← Kaggle download via kagglehub, or passthrough to a
    │   │                       local --data-path folder (Zindi/DrivenData).
    │   │
    │   ├── validate.py       ← Data loading (keeps categoricals), task detection,
    │   │                       frozen-fold creation/reuse, adversarial validation,
    │   │                       metric scoring.
    │   │
    │   ├── tuner.py          ← Optuna hyperparameter tuning for XGBoost, LightGBM,
    │   │                       CatBoost, with correct early stopping and the
    │   │                       winning trial's tree count carried into the final fit.
    │   │
    │   ├── train.py          ← Model trainers — default, tuned, and depth-1 variants
    │   │                       — all working on the frozen folds and native
    │   │                       categoricals, returning OOF + test predictions.
    │   │
    │   ├── features.py       ← Out-of-fold, smoothed target encoding; frequency
    │   │                       encoding; numeric interactions.
    │   │
    │   ├── ensemble.py       ← Rank averaging and Caruana-style hill climbing.
    │   │
    │   └── submit.py         ← Submission CSV generation + Kaggle submission/polling.
    │
    ├── state/
    │   ├── log.py            ← JSON logger. Reads/writes state/log.json.
    │   └── experiments.py    ← Save/load the OOF + test-prediction library used by
    │                           hill climbing.
    │
    └── kaggle_wrapper.ipynb  ← Jupyter notebook for the final Kaggle GPU run.
```

---

## FAQ

**Can I stop and resume?**
Yes. The agent reads `state/log.json` and `state/folds.json` at startup and continues from where it left off, on the same frozen folds. Delete both (and `state/experiments/`) for a clean start.

**I don't have a GPU. Will this work?**
Yes — the hardware detector reduces tree counts and disables GPU-specific settings automatically.

**I don't have a Kaggle account. Can I still use this?**
Yes — pass `--data-path` pointing at any folder with `train.csv`/`test.csv` and skip the submission step.

**What if my competition uses a metric not listed?**
Add it to `pipeline/validate.py`'s `cross_val_score` and it'll work everywhere the metric is threaded through (tuning, gating, hill climbing).

**Does it work with Zindi?**
Yes — see [Using with Zindi.africa](#using-with-zindiafrica).

**Can I add my own hypothesis?**
Add a function in `worker.py`, register it in both the `handlers` dict and `KNOWN_HYPOTHESES`, and add its name to `PHASE1_HYPOTHESES`/`PHASE2_HYPOTHESES` in `main.py`. A startup check validates the routing lists against the worker's registry, so a typo fails loudly at launch instead of silently skipping iterations.

**My competition has repeated entities (users, molecules, patients) or is time-ordered — will the default folds leak?**
The auto-picked splitter (`get_splitter` in `pipeline/validate.py`) only distinguishes classification from regression — it can't detect grouping or time order from the CSV alone. If your data has either property, modify `get_splitter` to use `GroupKFold` or `TimeSeriesSplit` before your first run; folds are frozen on first use, so this must happen before `state/folds.json` exists.

---

## Related

This tool is based on *The Kaggle Book* 2nd Ed. (Massaron, Tunguz, Banachewicz, Packt 2025), *Effective XGBoost* (2nd Ed.), the NVIDIA Kaggle Grandmasters' published playbook, and Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem* — synthesised into an autonomous research agent using the Karpathy-style skill pattern.

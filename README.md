# kaggle-research

An autonomous competition agent that runs Kaggle, Zindi, and DrivenData competitions for you. Implements a **research loop**: hypothesise → implement → CV verify → keep/revert → submit — grounded in the practices Kaggle Grandmasters actually use (frozen CV folds, noise-floor gating, out-of-fold target encoding, hill-climbing ensembles).

Supports **binary classification** and **regression** — auto-detected from your data. (Multiclass isn't supported yet; the pipeline tells you explicitly rather than producing silently wrong predictions.)

## Quick start

This project runs as a Python command-line program. An AI harness can set it up and operate it for you, but it needs access to a terminal, a writable project folder, Python dependencies, and Kaggle credentials. The repository does not host a web app.

### 1. Install the skill in a coding-agent harness

You need Node.js/npm for `npx skills`, Python 3.11 or later, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). Install the skill globally for detected supported agents:

```bash
npx skills add theAfricanQuant/kaggle-research --skill kaggle-research --global
```

Or target one harness explicitly:

```bash
npx skills add theAfricanQuant/kaggle-research --skill kaggle-research --global --agent codex
npx skills add theAfricanQuant/kaggle-research --skill kaggle-research --global --agent claude-code
npx skills add theAfricanQuant/kaggle-research --skill kaggle-research --global --agent opencode
```

Omit `--global` to install the skill into the current project so collaborators can share it. Add `--copy` if your environment does not support symlinks. The installer supports these and many other coding agents; see [`npx skills`](https://github.com/vercel-labs/skills) for its current agent list. This installs the workflow instructions, not Python, `uv`, or Kaggle credentials.

Open your competition project folder in the harness and ask:

> Use the `kaggle-research` skill to set up and run the Kaggle competition at `https://www.kaggle.com/competitions/titanic/overview`. Check the competition's Evaluation page and data rules, save the verified metric and validation setup in `competition.json`, run 10 iterations with 10 Optuna trials, and write a submission CSV. Do not submit it to the leaderboard.

The competition URL contains the slug (`titanic` in this example). The command accepts the slug, not the full URL. The agent should create a separate competition project, explain any metric or split uncertainty, install dependencies, and run the pipeline. Review the generated `competition.json` and the startup warnings before trusting the CV score.

### 2. Use it from ChatGPT

If your ChatGPT workspace has Skills enabled, download the repository and upload a skill package containing the root `SKILL.md` and the `template/` folder from **Skills → Create → Upload from your computer**. Skills are currently available only to eligible ChatGPT Business, Enterprise, Healthcare, and Edu workspaces, subject to admin settings; see [OpenAI's Skills in ChatGPT guide](https://help.openai.com/en/articles/20001066-skills-in-chatgpt).

A ChatGPT skill provides the workflow instructions. To actually download competition data, install Python packages, and run the pipeline, ChatGPT also needs a connected coding workspace or agent with terminal and filesystem access plus Kaggle credentials. In a regular chat without those tools, ask ChatGPT to help prepare the `competition.json` and commands, then run them in a terminal yourself. Never paste your Kaggle token into a prompt.

### 3. Connect your Kaggle account

Accept the competition rules on Kaggle first. Authenticate with one of Kaggle's supported methods: `kaggle auth login`, the `KAGGLE_API_TOKEN` environment variable, `~/.kaggle/access_token`, or the legacy `~/.kaggle/kaggle.json` credentials file. The terminal quick start below installs the `kaggle` CLI into the project; then run `uv run kaggle auth login` to sign in. See [Kaggle's CLI authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) and [kagglehub authentication](https://github.com/Kaggle/kagglehub#authenticate). Keep credentials out of the repository and prompts.

For Zindi or DrivenData, download `train.csv`, `test.csv`, and `sample_submission.csv` yourself; no Kaggle sign-in is needed. The data folder is passed with `--data-path /path/to/competition-data`; see [Using with Zindi](#using-with-zindiafrica).

### 4. Run it yourself in a terminal

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11+, then create a clean competition project from the template:

```bash
git clone https://github.com/theAfricanQuant/kaggle-research.git
cd kaggle-research/template
uv run main.py --name titanic --out .. --competition titanic
cd ../titanic
uv sync
uv run kaggle auth login
uv run main.py --competition titanic --iterations 10 --optuna-trials 10
```

The first command creates `kaggle-research/titanic/` without changing the reusable template. The run downloads Kaggle competition files, runs a short research pass, and writes `submission.csv` and usually `submission_final.csv`. It does **not** send a leaderboard submission. For local data instead, replace the competition slug with a label and add `--data-path /path/to/competition-data`.

### 5. Make a leaderboard submission only when ready

The safe default only creates CSV files. To authorize actual Kaggle submissions, pass `--submit` and choose a small budget, for example `--submit --max-submissions 1`. Kaggle submissions consume your competition submission allowance.

---

## What it does, step by step

### Step 1: Download the data

For Kaggle competitions, `kagglehub` downloads the train/test CSVs and caches them in `~/.cache/kagglehub/`. For Zindi/DrivenData (no download API — see [Using with Zindi](#using-with-zindiafrica)), pass `--data-path <folder>` pointing at your manually downloaded `train.csv`/`test.csv`.

### Step 2: Detect your hardware

Checks for an NVIDIA GPU with `nvidia-smi`, RAM, and CPU cores. Chooses conservative tree counts and enables CatBoost GPU training when a supported GPU is detected. PyTorch is not required.

### Step 3: Detect the task type and freeze the CV folds

Reads the target column and infers binary classification or regression; override with `--task` or `competition.json`. The optimization metric comes from `--metric`, then `competition.json`, then falls back to ROC-AUC (classification) or R² (regression). Always set and verify it against Kaggle's Evaluation page.

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
7. **Submit** — always writes valid CSVs from `sample_submission.csv`; sends them to Kaggle only when `--submit` explicitly authorizes it, while enforcing `--max-submissions` and skipping duplicate prediction hashes
8. **Track alignment** — after 5+ submissions, computes the Spearman rank correlation between CV and leaderboard scores; warns below 0.3

### Step 6: Final hill-climbing ensemble

Once all hypotheses are exhausted, the agent runs **cross-fitted Caruana-style hill climbing** over every persisted experiment. Ensemble members are selected inside meta-folds and scored on held-out rows; final test weights are then learned from all finite OOF rows. The blended test prediction is always written to `submission_final.csv` using the competition's real headers.

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
- **Kaggle account** and credentials (Kaggle competitions only) — authenticate with `uv run kaggle auth login`, or configure a Kaggle API token / legacy credentials file as described above.

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
| `--submit` | off | Explicitly authorize Kaggle leaderboard submissions |
| `--max-submissions` | 3 | Hard submission budget when `--submit` is enabled |
| `--cv-strategy` | `auto` | `stratified`, `kfold`, `group`, or forward-chaining `time` |
| `--group-col` | none | Entity column used for GroupKFold and excluded from features |
| `--time-col` | none | Ordering column used for temporal CV and excluded from features |
| `--n-splits` | 5 | Number of frozen CV folds |
| `--seed` | 42 | Fold-generation seed for shuffled CV |
| `--metric` | `auto` | Override the competition brief. Classification: `roc_auc`, `average_precision`, `logloss`, `accuracy`, `balanced_accuracy`, `f1`. Regression: `rmse`, `rmsle`, `mae`, `r2` |
| `--optuna-trials` | 50 | Hyperparameter trials per tuning session |
| `--task` | `auto` | Force task type: `classification`, `regression` |
| `--noise-seeds` | 3 | Seeds used to estimate the CV noise floor before the loop starts |

### Example: quick run to test things work

```bash
uv run main.py --competition "tabular-playground-series-jan-2021" \
  --iterations 10 --optuna-trials 10
```

### Competition brief (recommended)

Put a `competition.json` beside the competition CSVs (or in the project root) to make the target, supported metric, and validation assumptions explicit and reproducible:

```json
{
  "target_col": "target",
  "task": "classification",
  "metric": "average_precision",
  "cv_strategy": "group",
  "group_col": "customer_id",
  "notes": "One row per transaction; customers must not cross folds."
}
```

Supported task values are `classification` and `regression`; supported metrics are listed in the CLI table. CLI `--metric`, `--cv-strategy`, `--group-col`, and `--time-col` override the brief. The run prints structural clues and warns when row-wise CV or a default metric is being used. These checks guide review; they cannot infer the competition's rules reliably from CSVs alone. Confirm the metric, split scheme, and any group/time columns against the competition page before trusting CV.

Numeric missing values are left intact because the tree models handle them natively; this avoids estimating imputation medians from validation rows before folds are applied. The loop scores OOF predictions with the selected metric and uses each model library's matching early-stopping metric where available. For threshold metrics such as F1, early stopping uses a probability/ranking loss proxy; the final CV score remains the requested metric.

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
    │   ├── run.py            ← Data/config fingerprints and resume compatibility.
    │   └── experiments.py    ← Atomic OOF/test predictions plus metadata used by
    │                           hill climbing.
    ├── tests/                ← Regression tests for CV, state, features, ensembles,
    │                           routing, and submission safety.
    │
    └── kaggle_wrapper.ipynb  ← Jupyter notebook for the final Kaggle GPU run.
```

---

## FAQ

**Can I stop and resume?**
Yes. Iteration numbering, accepted feature transforms, diagnostics, folds, and experiment artifacts are restored. Data and run settings are fingerprinted; incompatible state is rejected instead of being silently reused. Move or delete the whole `state/` directory for a clean run.

**I don't have a GPU. Will this work?**
Yes — the hardware detector reduces tree counts and disables GPU-specific settings automatically.

**I don't have a Kaggle account. Can I still use this?**
Yes — pass `--data-path` pointing at any folder with `train.csv`/`test.csv` and skip the submission step.

**What if my competition uses a metric not listed?**
Add it to `pipeline/validate.py`'s `cross_val_score`, `metric_higher_is_better`, and the model-specific early-stopping mapping in `pipeline/train.py`. Then check that tuning, noise-floor gating, ensembling, submission selection, and reporting all handle its direction and prediction format correctly. Custom or unsupported Kaggle metrics must not be silently replaced with a default.

**Does it work with Zindi?**
Yes — see [Using with Zindi.africa](#using-with-zindiafrica).

**Can I add my own hypothesis?**
Add a function in `worker.py`, register it in both the `handlers` dict and `KNOWN_HYPOTHESES`, and add its name to `PHASE1_HYPOTHESES`/`PHASE2_HYPOTHESES` in `main.py`. A startup check validates the routing lists against the worker's registry, so a typo fails loudly at launch instead of silently skipping iterations.

**My competition has repeated entities (users, molecules, patients) or is time-ordered — will the default folds leak?**
The CSV alone may not reveal the correct split. Verify the competition rules and data collection process, then set `group_col` or `time_col` in `competition.json` (or use `--group-col` / `--time-col`) before the first run. Folds are frozen on first use; if the split strategy changes, start a fresh run by moving or deleting its `state/` directory.

---

## Related

This tool is based on *The Kaggle Book* 2nd Ed. (Massaron, Tunguz, Banachewicz, Packt 2025), *Effective XGBoost* (2nd Ed.), the NVIDIA Kaggle Grandmasters' published playbook, and Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem* — synthesised into an autonomous research agent using the Karpathy-style skill pattern.

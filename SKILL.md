---
name: kaggle-research
description: Runs an autonomous research loop for Kaggle, Zindi, and DrivenData tabular competitions — hypothesise, train with cross-validation, keep or revert based on CV, submit to the leaderboard. Use this skill whenever the user wants to enter, compete in, or iterate on a Kaggle or Zindi competition, mentions a competition slug or URL, asks to "find the best model/ensemble" for tabular data, wants automated hyperparameter tuning or ensembling for a competition, or says things like "compete for me", "run this competition", or "use the kaggle-research skill". Also use it for general tabular ML competition strategy questions (CV design, target encoding, ensembling, submission strategy) even if the user hasn't named a specific competition yet.
compatibility: Requires Python 3.11+, uv, and a Kaggle API token (~/.kaggle/kaggle.json) for Kaggle competitions. Zindi/DrivenData competitions require manually downloaded data.
---

# kaggle-research

Autonomous Kaggle/Zindi competition agent. Designed to run inside AI coding agent harnesses (opencode, Claude Code, pi.dev, Codex).

The loop: hypothesise → implement → CV verify → keep/revert → submit.

## What you need to do

The user is in an empty project folder. Your job is to:

1. Copy this skill's `template/` files into the current working directory
2. Install dependencies
3. Ensure the Kaggle API token is set up (Kaggle competitions only)
4. Scaffold a named project folder
5. Run the competition loop
6. Monitor progress and report back

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

If `~/.kaggle/kaggle.json` doesn't exist, ask the user to download it from kaggle.com/account → Create API Token and place it there. Zindi and DrivenData don't have this requirement — see "Using with Zindi" in README.md.

### 4. Scaffold a project folder

```bash
uv run main.py --competition "<slug>" --name <slug> --iterations 50
cd <slug>
```

### 5. Run the research loop

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

Routing is relative to the marginal gain of the last few hypotheses, not fixed absolute CV thresholds — a CV of 0.75 is a winning score in some competitions and a broken baseline in others. See `main.py`'s `route_next_hypothesis` for the exact logic.

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

Full methodology (CV design, target encoding, ensembling strategy, submission policy) is documented in `README.md` — read it before making changes to the loop.

## Extending

To add a new hypothesis: add a function in `worker.py` and map it in `_run_hypothesis`. The router in `main.py` auto-picks the next one based on CV.

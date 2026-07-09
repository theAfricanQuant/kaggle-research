import sys, argparse, logging
from datetime import datetime
from pathlib import Path

from hardware import detect_hardware

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kaggle-research")

METRICS_CLS = ["roc_auc", "logloss", "accuracy", "f1"]
METRICS_REG = ["rmse", "mae", "r2"]

# Baselines tried once per run, in order, before Optuna tuning kicks in.
PHASE1_HYPOTHESES = [
    "lgbm_defaults", "fe_target_encoding", "fe_frequency", "fe_interactions",
    "xgb_defaults", "catboost_defaults", "depth1_xgb_ensemble",
]
PHASE2_HYPOTHESES = ["optuna_xgb", "optuna_lgbm", "optuna_catboost"]


def _validate_hypothesis_names():
    from worker import KNOWN_HYPOTHESES
    unknown = set(PHASE1_HYPOTHESES + PHASE2_HYPOTHESES) - KNOWN_HYPOTHESES
    if unknown:
        raise RuntimeError(
            f"Routing lists reference hypotheses the worker doesn't implement: {sorted(unknown)}. "
            f"Register them in worker.py's handlers and KNOWN_HYPOTHESES."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--data-path", default=None,
                        help="Local folder with train.csv/test.csv (required for Zindi/DrivenData; "
                             "Kaggle competitions download automatically if omitted).")
    parser.add_argument("--name", default=None,
                        help="Project folder name. Creates and scaffolds a new folder if provided. "
                             "Default: run in-place.")
    parser.add_argument("--out", default=None,
                        help="Parent directory for --name (default: current directory).")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--submission-interval", type=int, default=5)
    parser.add_argument("--task", choices=["classification", "regression", "auto"], default="auto")
    parser.add_argument("--metric", default="auto",
                        help=f"Optimisation metric. Auto: roc_auc (cls) or r2 (reg). "
                             f"Classification: {METRICS_CLS}. Regression: {METRICS_REG}.")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Trials per Optuna study when tuning")
    parser.add_argument("--noise-seeds", type=int, default=3,
                        help="Seeds used to estimate the CV noise floor before the loop starts.")
    args = parser.parse_args()

    HERE = Path(__file__).parent.resolve()

    if args.name:
        _scaffold_project(args, HERE)
        return

    hw = detect_hardware()
    STATE_DIR = HERE / "state"

    from worker import run_hypothesis
    from pipeline.download import fetch_data
    from pipeline.validate import get_data, detect_task, load_or_create_folds, adversarial_validation_auc, cross_val_score, metric_higher_is_better
    from pipeline.submit import kaggle_submit, poll_for_score, save_submission_csv
    from pipeline.ensemble import hill_climb, apply_weights_to_test
    from state.log import load_state, save_state, LogEntry
    from state.experiments import save_experiment, load_experiment_library

    _validate_hypothesis_names()
    log.info(f"Hardware: GPU={hw['gpu']} ({hw['gpu_name']}) "
             f"RAM={hw['ram_gb']}GB Cores={hw['cores']} Kaggle env={hw['on_kaggle']}")

    data_path = fetch_data(args.competition, local_path=args.data_path)
    X, y, X_test, test_ids, cat_cols = get_data(data_path)

    task = detect_task(y) if args.task == "auto" else args.task
    log.info(f"Task: {task}" + (" (auto-detected)" if args.task == "auto" else ""))

    metric = args.metric if args.metric != "auto" else ("roc_auc" if task == "classification" else "r2")
    log.info(f"Optimising for: {metric}")

    folds = load_or_create_folds(STATE_DIR, X, y, task)
    log.info(f"Using {len(folds)} frozen CV folds (state/folds.json)")

    adv_auc = adversarial_validation_auc(X, X_test)
    if adv_auc is not None:
        level = log.warning if adv_auc > 0.7 else log.info
        level(f"Adversarial validation AUC (train vs test): {adv_auc:.3f}"
              + (" — train/test distributions differ; CV may not reflect the leaderboard" if adv_auc > 0.7 else ""))

    state = load_state(STATE_DIR / "log.json")
    state.setdefault("competition", args.competition)
    state.setdefault("task", task)
    state.setdefault("metric", metric)
    state.setdefault("iterations", [])
    state.setdefault("tried_hypotheses", [])
    save_state(STATE_DIR / "log.json", state)

    noise_floor = _estimate_noise_floor(X, y, X_test, hw, task, metric, folds, cat_cols, args.noise_seeds)
    log.info(f"CV noise floor (±1 std across {args.noise_seeds} seeds): {noise_floor:.4f} — "
             f"improvements smaller than this are treated as noise, not progress")

    feature_state = {"X": X, "X_test": X_test}
    higher_is_better = metric_higher_is_better(metric)

    for iteration in range(1, args.iterations + 1):
        log.info(f"=== Iteration {iteration}/{args.iterations} ===")

        hypothesis = route_next_hypothesis(state, task, iteration)
        if hypothesis is None:
            log.info("No untried hypotheses with expected marginal gain remain — stopping early")
            break
        log.info(f"Hypothesis: {hypothesis}")

        ctx = dict(y=y, hw=hw, task=task, metric=metric, cat_cols=cat_cols, folds=folds,
                   feature_state=feature_state, optuna_trials=args.optuna_trials)
        result = run_hypothesis(hypothesis, ctx)
        state["tried_hypotheses"].append(hypothesis)

        if result is None:
            save_state(STATE_DIR / "log.json", state)
            continue

        cv_before = state.get("latest_cv")
        delta = result["cv_score"] - (cv_before if cv_before is not None else 0)
        exp_path = save_experiment(STATE_DIR, f"{hypothesis}_{iteration}", result["oof"], result["test_preds"], result["cv_score"])
        entry = LogEntry(
            iteration=iteration, hypothesis=hypothesis,
            cv_before=cv_before, cv_after=result["cv_score"], delta=delta,
            experiment_path=exp_path,
            timestamp=datetime.now().isoformat(),
        )
        state["iterations"].append(entry.to_dict())

        improved = _beats_noise_floor(cv_before, result["cv_score"], higher_is_better, noise_floor)
        if improved:
            symbol = "✅" if cv_before is not None else "📋"
            log.info(f"{symbol} CV: {_fmt(cv_before)} → {_fmt(result['cv_score'])} (Δ{delta:+.4f}, "
                     f"exceeds noise floor {noise_floor:.4f})")
            state["latest_cv"] = result["cv_score"]
            if "candidate_features" in result:
                feature_state["X"], feature_state["X_test"] = result["candidate_features"]
                log.info(f"  Feature set from '{hypothesis}' kept — later hypotheses build on it")
        else:
            log.info(f"❌ Within noise floor (Δ{delta:+.4f} < {noise_floor:.4f}) — not adopted as new best, "
                     f"but experiment saved for ensembling")

        should_submit = iteration >= 10 and iteration % args.submission_interval == 0
        if should_submit:
            _submit_current_best(state, STATE_DIR, args, test_ids, task, data_path,
                                 kaggle_submit, poll_for_score, save_submission_csv, higher_is_better)

        save_state(STATE_DIR / "log.json", state)
        log.info(f"Iterations: {len(state['iterations'])} | Best CV: {_fmt(state.get('latest_cv'))}")

    log.info("=== Research loop complete — running final hill-climbing ensemble ===")
    oof_lib, test_lib, _ = load_experiment_library(STATE_DIR)
    if len(oof_lib) >= 2:
        weights, blended_oof, history = hill_climb(y, oof_lib, task, metric)
        blended_score = cross_val_score(y, blended_oof, task, metric)
        log.info(f"Hill-climbed ensemble ({len(history)} rounds): CV {blended_score:.4f}")
        log.info(f"Selection weights: {weights}")
        state["final_ensemble_weights"] = weights
        state["final_ensemble_cv"] = blended_score

        prev_best = state.get("latest_cv")
        ensemble_wins = (
            prev_best is None
            or (blended_score > prev_best if higher_is_better else blended_score < prev_best)
        )
        if ensemble_wins:
            state["latest_cv"] = blended_score
        # Always write the final-ensemble submission when test predictions exist:
        # hill climbing never scores below the best library member on OOF, and a
        # short run may not have produced any submission CSV at all yet.
        if test_lib:
            usable_weights = {n: w for n, w in weights.items() if n in test_lib}
            if usable_weights:
                final_test_preds = apply_weights_to_test(usable_weights, test_lib)
                sub_path = save_submission_csv(test_ids, final_test_preds, data_path=data_path,
                                               path=str(STATE_DIR.parent / "submission_final.csv"))
                state["final_submission_path"] = sub_path

    save_state(STATE_DIR / "log.json", state)
    log.info(f"Best CV: {_fmt(state.get('latest_cv'))} | Best LB: {state.get('last_lb')}")


def _scaffold_project(args, HERE):
    import shutil, subprocess
    parent = Path(args.out).resolve() if args.out else HERE.parent
    dest = parent / args.name
    if dest.exists():
        log.error(f"Folder already exists: {dest}")
        sys.exit(1)
    log.info(f"Scaffolding new project: {dest}")
    for item in HERE.iterdir():
        if item.name in (".git", "__pycache__"):
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name, ignore=lambda d, f: {"__pycache__"})
        else:
            shutil.copy2(item, dest / item.name)
    (dest / "pyproject.toml").write_text(
        (dest / "pyproject.toml").read_text().replace('name = "kaggle-research"', f'name = "{args.name}"')
    )
    (dest / ".python-version").write_text("3.12\n")
    (dest / "state" / "log.json").unlink(missing_ok=True)
    (dest / "state" / "folds.json").unlink(missing_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=dest)
    subprocess.run(["git", "add", "-A"], cwd=dest)
    subprocess.run(["git", "commit", "-q", "-m", f"initial: {args.name}"], cwd=dest)
    log.info(f"Project created. Switch to it and run:\n"
             f"  cd {dest}\n  uv sync\n"
             f"  uv run main.py --competition \"{args.competition}\" --iterations {args.iterations}")


def _estimate_noise_floor(X, y, X_test, hw, task, metric, folds, cat_cols, n_seeds):
    """Trains the cheapest baseline (LightGBM defaults) with several
    different fold-shuffle seeds to measure how much CV moves from
    randomness alone. Hypotheses must beat this to be considered real
    progress — otherwise the loop ratchets upward on noise, a documented
    failure mode of naive keep-if-better agents (Deotte's "plus or minus a
    little bit" seed variance; MLE-bench's "severe overfitting" finding).
    """
    from pipeline.train import train_lgbm
    from pipeline.validate import cross_val_score, get_splitter

    if n_seeds < 2:
        return 1e-4  # degenerate fallback; effectively disables the gate

    scores = []
    for seed in range(n_seeds):
        seeded_folds = [(tr, va) for tr, va in get_splitter(task, n_splits=len(folds), random_state=seed).split(X, y)]
        oof, _, _ = train_lgbm(X, y, None, hw, task, seeded_folds, cat_cols)
        scores.append(cross_val_score(y, oof, task, metric))
    import numpy as np
    return float(np.std(scores)) or 1e-4


def _beats_noise_floor(cv_before, cv_after, higher_is_better, noise_floor):
    if cv_before is None:
        return True
    delta = cv_after - cv_before
    return delta > noise_floor if higher_is_better else delta < -noise_floor


def route_next_hypothesis(state, task, iteration):
    """Picks the next untried hypothesis. Phase 1 (fast baselines) always
    runs first and in order, since it also builds the feature-engineering
    base that Phase 2 tunes on top of. Phase 2 (Optuna tuning) hypotheses
    are chosen by which model family hasn't been tried yet, not by
    absolute CV thresholds — a CV of 0.75 is a strong result in some
    competitions and a broken baseline in others, so routing on untried
    work rather than a fixed score table generalizes across competitions.
    """
    tried = set(state.get("tried_hypotheses", []))
    for hyp in PHASE1_HYPOTHESES:
        if hyp not in tried:
            return hyp
    for hyp in PHASE2_HYPOTHESES:
        if hyp not in tried:
            return hyp
    return None  # everything tried; final hill-climbing phase takes over


def _submit_current_best(state, STATE_DIR, args, test_ids, task, data_path,
                         kaggle_submit, poll_for_score, save_submission_csv, higher_is_better):
    from state.experiments import load_experiment_library
    oof_lib, test_lib, scores = load_experiment_library(STATE_DIR)
    if not scores:
        return
    # min() for rmse/logloss/mae — picking max there would submit the worst model
    pick = max if higher_is_better else min
    best_name = pick(scores, key=scores.get)
    if best_name not in test_lib:
        log.info("Skipping submission — no test predictions available for the current best experiment")
        return

    sub_path = save_submission_csv(test_ids, test_lib[best_name], data_path=data_path,
                                   path=str(STATE_DIR.parent / "submission.csv"))
    if args.data_path:
        log.info(f"Local-data competition — upload {sub_path} to the platform manually (no submission API)")
        return

    sub = kaggle_submit(sub_path, args.competition, f"iter {state['iterations'][-1]['iteration']}: {best_name[:60]}")
    lb_score = poll_for_score(sub, args.competition)
    log.info(f"Leaderboard: {lb_score} (CV: {_fmt(state.get('latest_cv'))})")
    state["iterations"][-1]["lb_score"] = lb_score
    state["last_lb"] = lb_score
    check_cv_lb_alignment(state)


def _fmt(v):
    return "—" if v is None else f"{v:.4f}"


def check_cv_lb_alignment(state):
    scores = [(it["cv_after"], it.get("lb_score")) for it in state["iterations"] if it.get("lb_score")]
    if len(scores) < 5:
        log.info(f"CV-LB alignment needs 5+ submissions to be meaningful ({len(scores)} so far)")
        return
    import numpy as np
    from scipy.stats import spearmanr
    cvs, lbs = zip(*scores)
    corr, _ = spearmanr(cvs, lbs)
    log.info(f"CV-LB rank correlation (last {len(scores)} submissions): {corr:.3f}")
    if abs(corr) < 0.3:
        log.warning("CV and LB poorly correlated — reconsider validation strategy "
                     "(check for train/test distribution shift, leakage, or wrong CV scheme)")


if __name__ == "__main__":
    main()

import sys, argparse, logging, hashlib, json
from datetime import datetime
from pathlib import Path

from hardware import detect_hardware

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kaggle-research")

METRICS_CLS = ["roc_auc", "average_precision", "logloss", "accuracy", "balanced_accuracy", "f1"]
METRICS_REG = ["rmse", "rmsle", "mae", "r2"]

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
    parser.add_argument("--cv-strategy", choices=["auto", "stratified", "kfold", "group", "time"],
                        default=None, help="Override cv_strategy from competition.json; auto uses configured split columns.")
    parser.add_argument("--group-col", default=None,
                        help="Entity/group column excluded from features and kept within one fold.")
    parser.add_argument("--time-col", default=None,
                        help="Temporal ordering column excluded from features; enables forward-chaining CV.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submit", action="store_true",
                        help="Authorize Kaggle leaderboard submissions. Without this flag only CSVs are written.")
    parser.add_argument("--max-submissions", type=int, default=3,
                        help="Maximum Kaggle submissions this run may make when --submit is present.")
    parser.add_argument("--metric", default=None,
                        help=f"Override metric from competition.json. Without either, defaults to roc_auc (cls) or r2 (reg). "
                             f"Classification: {METRICS_CLS}. Regression: {METRICS_REG}.")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Trials per Optuna study when tuning")
    parser.add_argument("--noise-seeds", type=int, default=3,
                        help="Seeds used to estimate the CV noise floor before the loop starts.")
    args = parser.parse_args()
    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    if args.iterations < 0:
        parser.error("--iterations cannot be negative")
    if args.optuna_trials < 1:
        parser.error("--optuna-trials must be positive")
    if args.submit and args.submission_interval < 1:
        parser.error("--submission-interval must be positive when --submit is used")
    if args.submit and args.max_submissions < 1:
        parser.error("--max-submissions must be positive when --submit is used")

    HERE = Path(__file__).parent.resolve()

    if args.name:
        _scaffold_project(args, HERE)
        return

    hw = detect_hardware()
    STATE_DIR = HERE / "state"

    from worker import run_hypothesis
    from pipeline.download import fetch_data
    from pipeline.validate import get_data, detect_task, load_or_create_folds, adversarial_validation_auc, cross_val_score, metric_higher_is_better, resolve_cv_strategy, validate_metric, validate_target, competition_diagnostics
    from pipeline.submit import kaggle_submit, poll_for_score, save_submission_csv
    from pipeline.ensemble import cross_fitted_hill_climb, apply_weights_to_test
    from state.log import load_state, save_state, LogEntry
    from state.experiments import save_experiment, load_experiment_library
    from state.run import fingerprint_data, initialize_or_validate_state

    _validate_hypothesis_names()
    log.info(f"Hardware: GPU={hw['gpu']} ({hw['gpu_name']}) "
             f"RAM={hw['ram_gb']}GB Cores={hw['cores']} Kaggle env={hw['on_kaggle']}")

    data_path = fetch_data(args.competition, local_path=args.data_path)
    # Optional checked-in competition brief makes metric/split assumptions
    # explicit and reproducible across runs. CLI flags take precedence.
    brief_path = Path(data_path) / "competition.json"
    if not brief_path.exists() and (HERE / "competition.json").exists():
        brief_path = HERE / "competition.json"
    brief = json.loads(brief_path.read_text()) if brief_path.exists() else {}
    if brief.get("notes"):
        log.info(f"Competition brief: {brief['notes']}")
    group_col = args.group_col or brief.get("group_col")
    time_col = args.time_col or brief.get("time_col")
    target_col = brief.get("target_col")
    X, y, X_test, test_ids, cat_cols, split_metadata = get_data(
        data_path, group_col=group_col, time_col=time_col, target_col=target_col)

    configured_task = brief.get("task", "auto")
    task_choice = args.task if args.task != "auto" else configured_task
    task = detect_task(y) if task_choice == "auto" else task_choice
    validate_target(task, y)
    log.info(f"Task: {task}" + (" (auto-detected)" if task_choice == "auto" else ""))

    metric_choice = args.metric or brief.get("metric", "auto")
    metric = validate_metric(task, metric_choice if metric_choice != "auto" else ("roc_auc" if task == "classification" else "r2"))
    log.info(f"Optimising for: {metric}")

    cv_strategy = resolve_cv_strategy(task, args.cv_strategy or brief.get("cv_strategy", "auto"),
                                      split_metadata["groups"], split_metadata["times"])
    for note in competition_diagnostics(X, y):
        log.warning(f"Competition review: {note}")
    if cv_strategy in ("stratified", "kfold"):
        log.warning("CV review: using row-wise CV. Confirm the competition does not require group, time, spatial, or predefined folds.")
    if metric_choice == "auto":
        log.warning(f"Metric review: defaulting to {metric}; confirm this matches the competition's exact evaluation metric.")
    data_fingerprint = fingerprint_data(data_path)
    run_config = {
        "competition": args.competition,
        "data_fingerprint": data_fingerprint,
        "task": task,
        "metric": metric,
        "target_col": target_col,
        "cv_strategy": cv_strategy,
        "group_col": group_col,
        "time_col": time_col,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "optuna_trials": args.optuna_trials,
        "noise_seeds": args.noise_seeds,
    }
    state = initialize_or_validate_state(load_state(STATE_DIR / "log.json"), run_config)
    folds = load_or_create_folds(
        STATE_DIR, X, y, task,
        groups=split_metadata["groups"], times=split_metadata["times"],
        strategy=cv_strategy, n_splits=args.n_splits, random_state=args.seed,
        fingerprint=state["run_fingerprint"],
    )
    log.info(f"Using {len(folds)} frozen {cv_strategy} CV folds (state/folds.json)")

    if "adversarial_validation_auc" in state:
        adv_auc = state["adversarial_validation_auc"]
    else:
        adv_auc = adversarial_validation_auc(X, X_test)
        state["adversarial_validation_auc"] = adv_auc
    if adv_auc is not None:
        level = log.warning if adv_auc > 0.7 else log.info
        level(f"Adversarial validation AUC (train vs test): {adv_auc:.3f}"
              + (" — train/test distributions differ; CV may not reflect the leaderboard" if adv_auc > 0.7 else ""))

    if "noise_floor" in state:
        noise_floor = state["noise_floor"]
    else:
        noise_floor = _estimate_noise_floor(X, y, X_test, hw, task, metric, folds, cat_cols,
                                            args.noise_seeds, cv_strategy)
        state["noise_floor"] = noise_floor
    log.info(f"CV noise floor (±1 std across {args.noise_seeds} seeds): {noise_floor:.4f} — "
             f"improvements smaller than this are treated as noise, not progress")

    feature_state = {"X": X, "X_test": X_test}
    accepted = state["accepted_feature_transforms"]
    if accepted:
        from pipeline.features import engineer_features
        accepted_X, accepted_X_test = engineer_features(X, y, X_test, cat_cols, accepted, folds)
        feature_state = {"X": accepted_X, "X_test": accepted_X_test}
        log.info(f"Reconstructed accepted feature state: {accepted}")

    higher_is_better = metric_higher_is_better(metric)
    save_state(STATE_DIR / "log.json", state)

    for iteration in range(state["next_iteration"], args.iterations + 1):
        log.info(f"=== Iteration {iteration}/{args.iterations} ===")

        hypothesis = route_next_hypothesis(state, task, iteration)
        if hypothesis is None:
            log.info("No untried hypotheses with expected marginal gain remain — stopping early")
            break
        log.info(f"Hypothesis: {hypothesis}")
        state["active_iteration"] = {"iteration": iteration, "hypothesis": hypothesis}
        save_state(STATE_DIR / "log.json", state)


        ctx = dict(y=y, hw=hw, task=task, metric=metric, cat_cols=cat_cols, folds=folds,
                   feature_state=feature_state, optuna_trials=args.optuna_trials)
        result = run_hypothesis(hypothesis, ctx)
        state["tried_hypotheses"].append(hypothesis)
        state["next_iteration"] = iteration + 1
        state.pop("active_iteration", None)

        if result is None:
            save_state(STATE_DIR / "log.json", state)
            continue

        cv_before = state.get("latest_cv")
        delta = result["cv_score"] - (cv_before if cv_before is not None else 0)
        experiment_metadata = {
            "iteration": iteration,
            "hypothesis": hypothesis,
            "task": task,
            "metric": metric,
            "cv_score": result["cv_score"],
            "run_fingerprint": state["run_fingerprint"],
            "best_params": result.get("best_params"),
            "best_iteration": result.get("best_iteration"),
            "proposed_transform": result.get("transform"),
            "accepted_feature_transforms": list(state["accepted_feature_transforms"]),
        }
        exp_path = save_experiment(
            STATE_DIR, f"{iteration:03d}_{hypothesis}", result["oof"], result["test_preds"],
            result["cv_score"], metadata=experiment_metadata,
        )
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
                transform = result.get("transform")
                if transform and transform not in state["accepted_feature_transforms"]:
                    state["accepted_feature_transforms"].append(transform)
                log.info(f"  Feature set from '{hypothesis}' kept — later hypotheses build on it")
        else:
            log.info(f"❌ Within noise floor (Δ{delta:+.4f} < {noise_floor:.4f}) — not adopted as new best, "
                     f"but experiment saved for ensembling")

        save_state(STATE_DIR / "log.json", state)
        should_submit = (
            args.submit
            and iteration >= 10
            and iteration % args.submission_interval == 0
            and len(state["submissions"]) < args.max_submissions
        )
        if should_submit:
            _submit_current_best(state, STATE_DIR, args, test_ids, task, data_path,
                                 kaggle_submit, poll_for_score, save_submission_csv, higher_is_better)
        save_state(STATE_DIR / "log.json", state)
        log.info(f"Iterations: {len(state['iterations'])} | Best CV: {_fmt(state.get('latest_cv'))}")

    log.info("=== Research loop complete — running final hill-climbing ensemble ===")
    oof_lib, test_lib, _ = load_experiment_library(STATE_DIR)
    if len(oof_lib) >= 2:
        weights, blended_oof, history = cross_fitted_hill_climb(y, oof_lib, task, metric)
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


def _estimate_noise_floor(X, y, X_test, hw, task, metric, folds, cat_cols, n_seeds, cv_strategy):
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
        if cv_strategy in ("group", "time"):
            seeded_folds = folds
        else:
            splitter = get_splitter(task, strategy=cv_strategy, n_splits=len(folds), random_state=seed)
            seeded_folds = [(tr, va) for tr, va in splitter.split(X, y)]
        oof, _, _ = train_lgbm(X, y, None, hw, task, seeded_folds, cat_cols, metric=metric)
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

    submission_hash = hashlib.sha256(test_lib[best_name].tobytes()).hexdigest()
    if submission_hash in state["submitted_hashes"]:
        log.info(f"Skipping duplicate submission for experiment {best_name}")
        return

    sub_path = save_submission_csv(test_ids, test_lib[best_name], data_path=data_path,
                                   path=str(STATE_DIR.parent / "submission.csv"))
    if args.data_path:
        log.info(f"Local-data competition — upload {sub_path} to the platform manually (no submission API)")
        return

    sub = kaggle_submit(sub_path, args.competition, f"iter {state['iterations'][-1]['iteration']}: {best_name[:60]}")
    if sub is None:
        return
    lb_score = poll_for_score(sub, args.competition)
    submitted_at = datetime.now().isoformat()
    state["submitted_hashes"].append(submission_hash)
    state["submissions"].append({
        "experiment": best_name,
        "prediction_hash": submission_hash,
        "cv_score": scores[best_name],
        "lb_score": lb_score,
        "submitted_at": submitted_at,
    })
    log.info(f"Leaderboard: {lb_score} (submitted CV: {_fmt(scores[best_name])})")
    state["last_lb"] = lb_score
    check_cv_lb_alignment(state)


def _fmt(v):
    return "—" if v is None else f"{v:.4f}"


def check_cv_lb_alignment(state):
    scores = [(sub["cv_score"], sub.get("lb_score")) for sub in state.get("submissions", []) if sub.get("lb_score") is not None]
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

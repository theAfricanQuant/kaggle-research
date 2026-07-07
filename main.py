import os, sys, json, argparse, logging
from datetime import datetime
from pathlib import Path

from hardware import detect_hardware

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kaggle-research")

METRICS_CLS = ["roc_auc", "logloss", "accuracy", "f1"]
METRICS_REG = ["rmse", "mae", "r2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
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
    args = parser.parse_args()

    HERE = Path(__file__).parent.resolve()

    # Scaffold a named project folder if --name is given
    if args.name:
        parent = Path(args.out).resolve() if args.out else HERE
        dest = parent / args.name
        if dest.exists():
            log.error(f"Folder already exists: {dest}")
            sys.exit(1)
        log.info(f"Scaffolding new project: {dest}")
        import shutil, subprocess
        for item in HERE.iterdir():
            if item.name in (".git", "__pycache__"):
                continue
            if item.is_dir():
                shutil.copytree(item, dest / item.name,
                                ignore=lambda d, f: {"__pycache__"})
            else:
                shutil.copy2(item, dest / item.name)
        (dest / "pyproject.toml").write_text(
            (dest / "pyproject.toml").read_text().replace(
                'name = "kaggle-research"', f'name = "{args.name}"'
            )
        )
        (dest / ".python-version").write_text("3.12\n")
        # Remove stale state log
        (dest / "state" / "log.json").unlink(missing_ok=True)
        # Init git
        subprocess.run(["git", "init", "-q"], cwd=dest)
        subprocess.run(["git", "add", "-A"], cwd=dest)
        subprocess.run(["git", "commit", "-q", "-m", f"initial: {args.name}"], cwd=dest)
        log.info(f"Project created. Switch to it and run:\n"
                 f"  cd {dest}\n"
                 f"  uv sync\n"
                 f"  uv run main.py --competition \"{args.competition}\" --iterations {args.iterations}")
        return  # exit so user enters the folder and runs

    hw = detect_hardware()
    STATE_DIR = HERE / "state"

    from worker import dispatch_workers
    from pipeline.download import fetch_data
    from pipeline.validate import get_data, detect_task
    from pipeline.submit import kaggle_submit, poll_for_score
    from state.log import load_state, save_state, LogEntry
    log.info(f"Hardware: GPU={hw['gpu']} ({hw['gpu_name']}) "
             f"RAM={hw['ram_gb']}GB Cores={hw['cores']} "
             f"Kaggle env={hw['on_kaggle']}")

    data_path = fetch_data(args.competition)

    if args.task == "auto":
        _, y_sample, _ = get_data(data_path)
        task = detect_task(y_sample)
        log.info(f"Auto-detected task: {task}")
    else:
        task = args.task
        log.info(f"Task: {task}")

    metric = args.metric
    if metric == "auto":
        metric = "roc_auc" if task == "classification" else "r2"
    log.info(f"Optimising for: {metric}")

    state = load_state(STATE_DIR / "log.json")
    state.setdefault("competition", args.competition)
    state.setdefault("task", task)
    state.setdefault("metric", metric)
    state.setdefault("iterations", [])
    save_state(STATE_DIR / "log.json", state)

    for iteration in range(1, args.iterations + 1):
        log.info(f"=== Iteration {iteration}/{args.iterations} ===")

        hypothesis = route_next_hypothesis(state, task, args.optuna_trials)
        log.info(f"Hypothesis: {hypothesis}")

        results = dispatch_workers(
            [hypothesis], data_path, hw, task,
            metric=metric, optuna_trials=args.optuna_trials
        )
        if not results:
            continue

        best = _pick_best(results, metric, task)
        entry = LogEntry(
            iteration=iteration,
            hypothesis=hypothesis,
            cv_before=state.get("latest_cv"),
            cv_after=best["cv_score"],
            delta=best["cv_score"] - (state.get("latest_cv") or 0),
            model_path=best.get("model_path"),
            preds_path=best.get("preds_path"),
            timestamp=datetime.now().isoformat(),
        )
        state["iterations"].append(entry.to_dict())

        _apply_improvement(state, entry)

        should_submit = iteration >= 10 and iteration % args.submission_interval == 0
        if should_submit:
            sub = kaggle_submit(best["preds_path"], f"iter {iteration}: {hypothesis[:60]}")
            lb_score = poll_for_score(sub, args.competition)
            log.info(f"Leaderboard: {lb_score} (CV: {_fmt(state['latest_cv'])})")
            entry.lb_score = lb_score
            state["last_lb"] = lb_score
            check_cv_lb_alignment(state)

        save_state(STATE_DIR / "log.json", state)
        log.info(f"Iterations: {len(state['iterations'])} | Best CV: {_fmt(state.get('latest_cv'))}")

    log.info("=== Autoresearch complete ===")
    log.info(f"Best CV: {_fmt(state.get('latest_cv'))} | Best LB: {state.get('last_lb')}")


def route_next_hypothesis(state, task, optuna_trials):
    if "latest_cv" not in state or state["latest_cv"] is None:
        return "stratified_5fold_lgbm_defaults"

    cv = state["latest_cv"]
    n = len(state["iterations"])
    cls = task == "classification"

    thresholds = {
        "fe_target_encoding": (None, 0.50) if not cls else (None, 0.70),
        "fe_interactions":    (0.50, 0.55) if not cls else (0.70, 0.75),
        "optuna_xgb":         (0.55, 0.65) if not cls else (0.75, 0.82),
        "optuna_lgbm":        (0.65, 0.70) if not cls else (0.82, 0.85),
        "optuna_catboost":    (0.70, 0.75) if not cls else (0.85, 0.87),
        "depth1_xgb":         None,
        "average":            None,
        "blend":              None,
        "stack":              None,
    }

    if n < 5:
        return "feature_engineering_target_encoding"
    elif n < 10:
        return "feature_engineering_interactions"

    for hyp, (lo, hi) in thresholds.items():
        if lo is None and hi is None:
            continue
        if lo is None:
            if cv < hi:
                return hyp
        elif hi is None:
            if cv >= lo:
                return hyp
        else:
            if lo <= cv < hi:
                return hyp

    if cv >= (0.75 if not cls else 0.87):
        return "depth1_xgb_ensemble"
    elif cv >= (0.78 if not cls else 0.88):
        return "average_lgbm_xgb_catboost"
    elif cv >= (0.82 if not cls else 0.9):
        return "blend_with_meta_model"
    else:
        return "stack_ensemble_all_top_models"


def _pick_best(results, metric, task):
    higher_is_better = metric in ("roc_auc", "r2", "accuracy", "f1")
    key = (lambda r: r["cv_score"]) if higher_is_better else (lambda r: -r["cv_score"])
    return max(results, key=key)


def _apply_improvement(state, entry):
    higher_is_better = state.get("metric", "roc_auc") in ("roc_auc", "r2", "accuracy", "f1")
    prev = state.get("latest_cv")

    if prev is None:
        improved = True
    elif higher_is_better:
        improved = entry.delta > 1e-4
    else:
        improved = entry.delta < -1e-4

    if improved:
        symbol = "✅" if entry.cv_before is not None else "📋"
        log.info(f"{symbol} CV: {_fmt(entry.cv_before)} → {_fmt(entry.cv_after)} (Δ{entry.delta:+.4f})")
        state["latest_cv"] = entry.cv_after
        state["best_model_path"] = entry.model_path
    else:
        log.info(f"❌ No improvement (Δ{entry.delta:+.4f}) — reverted")
        if entry.cv_before is not None:
            state["latest_cv"] = entry.cv_before


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def check_cv_lb_alignment(state):
    scores = [(it["cv_after"], it.get("lb_score")) for it in state["iterations"] if it.get("lb_score")]
    if len(scores) < 3:
        return
    import numpy as np
    cvs, lbs = zip(*scores)
    corr = np.corrcoef(cvs, lbs)[0, 1]
    log.info(f"CV-LB correlation (last {len(scores)} submissions): {corr:.3f}")
    if abs(corr) < 0.3:
        log.warning("CV and LB poorly correlated — reconsider validation strategy")


if __name__ == "__main__":
    main()

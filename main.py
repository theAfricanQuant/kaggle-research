import os, sys, json, time, argparse, logging
from datetime import datetime
from pathlib import Path

from hardware import detect_hardware
from worker import dispatch_workers
from pipeline.download import fetch_data
from pipeline.validate import cross_val_score, get_data, detect_task
from pipeline.submit import kaggle_submit, poll_for_score
from state.log import load_state, save_state, LogEntry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kaggle-research")

HERE = Path(__file__).parent
STATE_DIR = HERE / "state"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--submission-interval", type=int, default=5)
    parser.add_argument("--final-days", type=int, default=3)
    parser.add_argument("--task", choices=["classification", "regression", "auto"], default="auto")
    args = parser.parse_args()

    hw = detect_hardware()
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

    state = load_state(STATE_DIR / "log.json")
    state.setdefault("competition", args.competition)
    state.setdefault("task", task)
    state.setdefault("iterations", [])
    save_state(STATE_DIR / "log.json", state)

    day_zero = datetime.now()

    for iteration in range(1, args.iterations + 1):
        log.info(f"=== Iteration {iteration}/{args.iterations} ===")

        hypothesis = route_next_hypothesis(state, task)
        log.info(f"Hypothesis: {hypothesis}")

        results = dispatch_workers([hypothesis], data_path, hw, task)
        if not results:
            continue

        best = max(results, key=lambda r: r["cv_score"])
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

        if state.get("latest_cv") is None or entry.delta > 1e-4:
            symbol = "✅" if entry.delta > 1e-4 else "📋"
            log.info(f"{symbol} CV: {_fmt(entry.cv_before)} → {_fmt(entry.cv_after)} (Δ{entry.delta:+.4f})")
            state["latest_cv"] = entry.cv_after
            state["best_model_path"] = best.get("model_path")
        else:
            log.info(f"❌ No improvement (Δ{entry.delta:+.4f}) — reverted")
            if entry.cv_before is not None:
                state["latest_cv"] = entry.cv_before

        should_submit = (
            iteration >= 10 and iteration % args.submission_interval == 0
        )
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


def route_next_hypothesis(state, task):
    if "latest_cv" not in state or state["latest_cv"] is None:
        return "stratified_5fold_lgbm_defaults"

    cv = state["latest_cv"]
    n = len(state["iterations"])

    if task == "classification":
        if n < 5:
            return "feature_engineering_target_encoding"
        elif cv < 0.7:
            return "feature_engineering_interactions"
        elif cv < 0.8:
            return "try_xgboost_with_tuning"
        elif cv < 0.85:
            return "try_catboost_with_tuning"
        elif cv < 0.88:
            return "average_lgbm_xgb_catboost"
        elif cv < 0.9:
            return "blend_with_meta_model"
        else:
            return "stack_ensemble_all_top_models"
    else:
        if n < 5:
            return "feature_engineering_target_encoding"
        elif cv < 0.5:
            return "feature_engineering_interactions"
        elif cv < 0.65:
            return "try_xgboost_with_tuning"
        elif cv < 0.75:
            return "try_catboost_with_tuning"
        elif cv < 0.8:
            return "average_lgbm_xgb_catboost"
        elif cv < 0.85:
            return "blend_with_meta_model"
        else:
            return "stack_ensemble_all_top_models"


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

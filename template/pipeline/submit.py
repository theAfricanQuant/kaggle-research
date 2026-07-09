import time, logging
import pandas as pd

log = logging.getLogger("kaggle-research")


def save_submission_csv(test_ids, predictions, target_col="target", path="submission.csv"):
    df = pd.DataFrame({"id": test_ids, target_col: predictions})
    df.to_csv(path, index=False)
    log.info(f"Submission saved to {path}")
    return path


def kaggle_submit(preds_path, message):
    """Submits to Kaggle via kagglehub. Returns None (no-op, logged) when
    there's nothing to submit or the competition isn't hosted on Kaggle —
    Zindi/DrivenData have no submission API, so save_submission_csv's
    output is meant for manual upload in that case (see README).
    """
    import kagglehub
    if preds_path is None:
        log.warning("No predictions to submit")
        return None
    log.info(f"Submitting: {message}")
    return kagglehub.competition_submit(preds_path, message)


def poll_for_score(submission, competition, max_retries=10, delay=30):
    import kagglehub
    if submission is None:
        return None
    for _ in range(max_retries):
        try:
            result = kagglehub.competition_submission_status(competition)
            score = getattr(result, "score", None) or (result.get("score") if isinstance(result, dict) else None)
            if score is not None:
                return float(score)
        except Exception:
            pass
        time.sleep(delay)
    return None

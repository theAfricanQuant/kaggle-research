import kagglehub, time, logging

log = logging.getLogger("kaggle-research")


def kaggle_submit(preds_path, message):
    if preds_path is None:
        log.warning("No predictions to submit")
        return None
    log.info(f"Submitting: {message}")
    competition = kagglehub.competition_submit(preds_path, message)
    return competition


def poll_for_score(submission, competition, max_retries=10, delay=30):
    if submission is None:
        return None
    for _ in range(max_retries):
        try:
            result = kagglehub.competition_submission_status(competition)
            score = getattr(result, "score", None) or result.get("score")
            if score is not None:
                return float(score)
        except Exception:
            pass
        time.sleep(delay)
    return None


def save_submission_csv(predictions, ids, path="submission.csv"):
    import pandas as pd
    df = pd.DataFrame({"id": ids, "target": predictions})
    df.to_csv(path, index=False)
    log.info(f"Submission saved to {path}")
    return path

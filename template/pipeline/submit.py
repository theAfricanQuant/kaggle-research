import os, csv, io, time, subprocess, logging
import pandas as pd

log = logging.getLogger("kaggle-research")


def save_submission_csv(test_ids, predictions, data_path=None, path="submission.csv"):
    """Writes a submission CSV using the competition's own column names.

    Kaggle validates headers, so 'id,target' only works by luck — when the
    competition ships a sample_submission.csv we copy its exact column
    names (and its id values as a fallback when we couldn't identify an id
    column in test.csv).
    """
    id_name, target_name = "id", "target"
    sample = None
    if data_path:
        sample_path = os.path.join(data_path, "sample_submission.csv")
        if os.path.exists(sample_path):
            sample = pd.read_csv(sample_path)
            id_name, target_name = sample.columns[0], sample.columns[1]

    if test_ids is None and sample is not None:
        test_ids = sample[id_name].values
    if test_ids is None:
        log.warning("No id column found in test.csv and no sample_submission.csv — "
                    "using a 0-based range, which most platforms will reject")
        test_ids = range(len(predictions))

    df = pd.DataFrame({id_name: test_ids, target_name: predictions})
    df.to_csv(path, index=False)
    log.info(f"Submission saved to {path} (columns: {id_name},{target_name})")
    return path


def kaggle_submit(preds_path, competition, message):
    """Submits via the official `kaggle` CLI — kagglehub can download
    competition data but has no submission API. The CLI reads the same
    ~/.kaggle/kaggle.json token the rest of the pipeline assumes.
    """
    if preds_path is None:
        log.warning("No predictions to submit")
        return None
    log.info(f"Submitting: {message}")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition,
             "-f", preds_path, "-m", message],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            log.warning(f"Submission failed: {result.stderr.strip() or result.stdout.strip()}")
            return None
        return competition
    except FileNotFoundError:
        log.warning("`kaggle` CLI not found — run `uv sync` (it's a project dependency) "
                    "or submit the CSV manually")
        return None
    except subprocess.TimeoutExpired:
        log.warning("Submission timed out")
        return None


def poll_for_score(submission, competition, max_retries=10, delay=30):
    """Polls `kaggle competitions submissions` until the newest submission
    has a public score (or gives up). Returns the score as float, or None.
    """
    if submission is None:
        return None
    for _ in range(max_retries):
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                rows = list(csv.DictReader(io.StringIO(result.stdout)))
                if rows:
                    latest = rows[0]
                    score = latest.get("publicScore") or latest.get("public_score")
                    status = (latest.get("status") or "").lower()
                    if score not in (None, "", "None"):
                        return float(score)
                    if "error" in status:
                        log.warning(f"Submission errored on Kaggle: {latest}")
                        return None
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        time.sleep(delay)
    return None

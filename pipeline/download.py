import os, logging
import kagglehub

log = logging.getLogger("kaggle-research")

DATA_CACHE = os.path.expanduser("~/.cache/kagglehub")


def fetch_data(competition):
    log.info(f"Downloading {competition}...")
    path = kagglehub.competition_download(competition)
    log.info(f"Data cached at: {path}")
    return path


def get_data_paths(data_path):
    files = []
    for root, dirs, fnames in os.walk(data_path):
        for f in fnames:
            if f.endswith(".csv") or f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    return files

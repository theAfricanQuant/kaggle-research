import os, logging

log = logging.getLogger("kaggle-research")


def fetch_data(competition, local_path=None):
    """Downloads via kagglehub for Kaggle competitions, or returns
    local_path unchanged for Zindi/DrivenData — those platforms have no
    download API, so the user places train.csv/test.csv in a folder
    themselves and passes --data-path.
    """
    if local_path:
        log.info(f"Using local data: {local_path}")
        return local_path
    import kagglehub
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

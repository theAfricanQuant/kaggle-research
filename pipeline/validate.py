import os, pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score


def detect_task(y):
    if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 20:
        return "classification"
    if np.issubdtype(y.dtype, np.floating):
        if len(np.unique(y)) <= 10:
            return "classification"
        return "regression"
    if y.dtype == bool or y.dtype == object:
        return "classification"
    return "regression"


def get_data(data_path, sample_frac=1.0):
    train_path = os.path.join(data_path, "train.csv")
    df = pd.read_csv(train_path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    target_col = "target" if "target" in df.columns else [c for c in df.columns if c not in ["id"]][-1]
    id_col = "id" if "id" in df.columns else None

    y = df[target_col].values
    X = df.drop(columns=[c for c in [target_col, id_col] if c and c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(-1)
    task = detect_task(y)
    return X.values, y, task


def get_splitter(task, n_splits=5, random_state=42):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_val_score(y_true, y_pred, task):
    if task == "classification":
        return roc_auc_score(y_true, y_pred)
    return r2_score(y_true, y_pred)

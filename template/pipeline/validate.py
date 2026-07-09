import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import roc_auc_score, r2_score, log_loss, mean_squared_error, mean_absolute_error, accuracy_score, f1_score


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


def _load_csv(data_path, name):
    path = os.path.join(data_path, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def get_data(data_path, sample_frac=1.0):
    """Loads train (and test, if present) preserving categorical columns.

    Returns (X_train_df, y, X_test_df_or_None, test_ids_or_None, cat_cols).
    Categoricals are kept as pandas 'category' dtype so CatBoost/LightGBM can
    use them natively instead of every column being coerced to numeric.
    """
    df = _load_csv(data_path, "train.csv")
    if df is None:
        raise FileNotFoundError(f"No train.csv found in {data_path}")
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    target_col = "target" if "target" in df.columns else [c for c in df.columns if c not in ("id",)][-1]
    id_col = "id" if "id" in df.columns else None

    y = df[target_col].values
    X = df.drop(columns=[c for c in [target_col, id_col] if c and c in df.columns])

    test_df = _load_csv(data_path, "test.csv")
    test_ids, X_test = None, None
    if test_df is not None:
        test_ids = test_df[id_col].values if id_col and id_col in test_df.columns else np.arange(len(test_df))
        X_test = test_df.drop(columns=[c for c in [id_col] if c and c in test_df.columns])
        X_test = X_test.reindex(columns=X.columns)

    cat_cols = []
    for col in X.columns:
        is_textlike = X[col].dtype == object or X[col].dtype.name in ("category", "str", "string")
        if is_textlike or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
            if X_test is not None:
                X_test[col] = X_test[col].astype(pd.CategoricalDtype(categories=X[col].cat.categories))
            cat_cols.append(col)
        else:
            X[col] = X[col].fillna(X[col].median())
            if X_test is not None:
                X_test[col] = X_test[col].fillna(X[col].median())

    return X, y, X_test, test_ids, cat_cols


def get_splitter(task, groups=None, n_splits=5, random_state=42):
    """Picks the CV scheme appropriate to the data structure.

    Grouped data (repeated entities) uses GroupKFold so no entity leaks
    across train/validation. Otherwise stratify classification targets;
    plain shuffled KFold for regression. Time-ordered competitions need
    TimeSeriesSplit, which this auto-picker can't detect from data alone —
    pass --group-col if your competition has repeated entities, and treat
    a temporal target column as a signal to switch splitters manually.
    """
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def load_or_create_folds(state_dir, X, y, task, groups=None, n_splits=5, random_state=42):
    """Freezes fold indices to disk on first call; every later hypothesis
    reuses the exact same folds so their OOF scores are comparable and
    stackable. Re-running with different folds each iteration silently
    invalidates every ensembling step downstream.
    """
    folds_path = Path(state_dir) / "folds.json"
    if folds_path.exists():
        with open(folds_path) as f:
            raw = json.load(f)
        return [(np.array(tr), np.array(va)) for tr, va in raw]

    splitter = get_splitter(task, groups=groups, n_splits=n_splits, random_state=random_state)
    split_args = (X, y, groups) if groups is not None else (X, y)
    folds = [(tr.tolist(), va.tolist()) for tr, va in splitter.split(*split_args)]
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_path, "w") as f:
        json.dump(folds, f)
    return [(np.array(tr), np.array(va)) for tr, va in folds]


def adversarial_validation_auc(X_train, X_test, n_splits=5):
    """Trains a classifier to distinguish train rows from test rows.
    AUC near 0.5 means the CV split can be trusted to reflect the test
    distribution; AUC far above 0.5 means train/test differ and CV scores
    may not transfer to the leaderboard.
    """
    import lightgbm as lgb
    if X_test is None or len(X_test) == 0:
        return None

    X_train_num = X_train.select_dtypes(include=[np.number]).fillna(-1)
    X_test_num = X_test.select_dtypes(include=[np.number]).fillna(-1)
    common = [c for c in X_train_num.columns if c in X_test_num.columns]
    if not common:
        return None

    Xa = pd.concat([X_train_num[common], X_test_num[common]], axis=0, ignore_index=True)
    ya = np.concatenate([np.zeros(len(X_train_num)), np.ones(len(X_test_num))])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(Xa))
    for tr, va in skf.split(Xa, ya):
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                    random_state=42, verbose=-1)
        model.fit(Xa.values[tr], ya[tr])
        oof[va] = model.predict_proba(Xa.values[va])[:, 1]
    return roc_auc_score(ya, oof)


def cross_val_score(y_true, y_pred, task, metric=None):
    metric = metric or ("roc_auc" if task == "classification" else "r2")
    if metric == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    if metric == "logloss":
        return log_loss(y_true, y_pred)
    if metric == "rmse":
        return mean_squared_error(y_true, y_pred) ** 0.5
    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric == "r2":
        return r2_score(y_true, y_pred)
    if metric == "accuracy":
        return accuracy_score(y_true, (y_pred > 0.5).astype(int))
    if metric == "f1":
        return f1_score(y_true, (y_pred > 0.5).astype(int))
    return roc_auc_score(y_true, y_pred) if task == "classification" else r2_score(y_true, y_pred)


def metric_higher_is_better(metric):
    return metric in ("roc_auc", "r2", "accuracy", "f1")

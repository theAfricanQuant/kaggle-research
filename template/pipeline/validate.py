import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             r2_score, log_loss, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score)


def detect_task(y):
    y = np.asarray(y)
    unique = np.unique(y)
    if y.dtype == bool or y.dtype == object or np.issubdtype(y.dtype, np.str_):
        return "classification"
    if len(unique) <= 2:
        return "classification"
    if np.issubdtype(y.dtype, np.integer):
        class_ratio = len(unique) / max(len(y), 1)
        if len(unique) <= 20 and class_ratio <= 0.2:
            return "classification"
    return "regression"


def validate_target(task, y):
    """Reject unsupported multiclass targets after any explicit task override."""
    n_classes = len(np.unique(y))
    if task == "classification" and n_classes != 2:
        raise ValueError(
            f"Classification target has {n_classes} classes; this pipeline currently "
            "supports binary classification only. Use --task regression for a continuous "
            "numeric target."
        )
    return task


def _load_csv(data_path, name):
    path = os.path.join(data_path, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def get_data(data_path, sample_frac=1.0, group_col=None, time_col=None, target_col=None):
    """Loads train (and test, if present) preserving categorical columns.

    Returns (X_train, y, X_test, test_ids, categorical_columns, split_metadata).
    Categoricals are kept as pandas 'category' dtype so CatBoost/LightGBM can
    use them natively instead of every column being coerced to numeric.
    """
    df = _load_csv(data_path, "train.csv")
    if df is None:
        raise FileNotFoundError(f"No train.csv found in {data_path}")
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    # sample_submission.csv names the real id and target columns — prefer it
    # over guessing, since competitions use anything from "Id" to "SalePrice".
    sample = _load_csv(data_path, "sample_submission.csv")
    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    inferred_target = None
    if sample is not None and len(sample.columns) >= 2:
        if sample.columns[0] in df.columns:
            id_col = sample.columns[0]
        if sample.columns[1] in df.columns:
            inferred_target = sample.columns[1]
    target_col = target_col or inferred_target
    if target_col is None:
        target_col = "target" if "target" in df.columns else [c for c in df.columns if c != id_col][-1]
    if target_col not in df.columns:
        raise KeyError(f"Configured target column '{target_col}' is missing from train.csv")

    for split_col, label in ((group_col, "group"), (time_col, "time")):
        if split_col and split_col not in df.columns:
            raise KeyError(f"Configured {label} column '{split_col}' is missing from train.csv")
    if time_col:
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    y = df[target_col].values
    groups = df[group_col].to_numpy() if group_col else None
    times = df[time_col].to_numpy() if time_col else None
    X = df.drop(columns=[c for c in [target_col, id_col, group_col, time_col] if c and c in df.columns])

    test_df = _load_csv(data_path, "test.csv")
    test_ids, X_test = None, None
    if test_df is not None:
        test_ids = test_df[id_col].values if id_col and id_col in test_df.columns else np.arange(len(test_df))
        X_test = test_df.drop(columns=[c for c in [id_col, group_col, time_col] if c and c in test_df.columns])
        X_test = X_test.reindex(columns=X.columns)

    cat_cols = []
    for col in X.columns:
        is_textlike = X[col].dtype == object or X[col].dtype.name in ("category", "str", "string")
        if is_textlike or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
            if X_test is not None:
                known_categories = X[col].cat.categories
                X_test[col] = X_test[col].where(X_test[col].isin(known_categories)).astype(pd.CategoricalDtype(categories=known_categories))
            cat_cols.append(col)
        # The tree models used here handle numeric NaNs natively. Avoid
        # full-data imputation, which would let validation rows affect medians.

    return X, y, X_test, test_ids, cat_cols, {"groups": groups, "times": times}


def resolve_cv_strategy(task, strategy="auto", groups=None, times=None):
    valid = {"auto", "stratified", "kfold", "group", "time"}
    if strategy not in valid:
        raise ValueError(f"Unknown CV strategy '{strategy}'. Choose from {sorted(valid)}")
    if strategy == "auto":
        if times is not None:
            return "time"
        if groups is not None:
            return "group"
        return "stratified" if task == "classification" else "kfold"
    if strategy == "group" and groups is None:
        raise ValueError("Group CV requires --group-col")
    if strategy == "time" and times is None:
        raise ValueError("Time CV requires --time-col")
    return strategy


def get_splitter(task, groups=None, times=None, strategy="auto", n_splits=5, random_state=42):
    """Picks the CV scheme appropriate to the data structure.

    Grouped data (repeated entities) uses GroupKFold so no entity leaks
    across train/validation. Otherwise stratify classification targets;
    plain shuffled KFold for regression. Time-ordered competitions need
    TimeSeriesSplit, which this auto-picker can't detect from data alone —
    pass --group-col if your competition has repeated entities, and treat
    a temporal target column as a signal to switch splitters manually.
    """
    strategy = resolve_cv_strategy(task, strategy, groups, times)
    if strategy == "group":
        return GroupKFold(n_splits=n_splits)
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def load_or_create_folds(state_dir, X, y, task, groups=None, times=None, strategy="auto",
                         n_splits=5, random_state=42, fingerprint=None):
    """Freezes fold indices to disk on first call; every later hypothesis
    reuses the exact same folds so their OOF scores are comparable and
    stackable. Re-running with different folds each iteration silently
    invalidates every ensembling step downstream.
    """
    folds_path = Path(state_dir) / "folds.json"
    resolved = resolve_cv_strategy(task, strategy, groups, times)
    if folds_path.exists():
        with open(folds_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "folds" not in payload:
            raise RuntimeError("Legacy folds.json is unsafe to reuse; move or delete state/ and rerun")
        if fingerprint and payload.get("fingerprint") != fingerprint:
            raise RuntimeError("Frozen folds do not match the current data/CV configuration")
        if payload.get("n_rows") != len(X):
            raise RuntimeError("Frozen folds have a different row count from the current training data")
        return [(np.array(tr), np.array(va)) for tr, va in payload["folds"]]

    splitter = get_splitter(task, groups, times, resolved, n_splits, random_state)
    if resolved == "time":
        order = np.argsort(np.asarray(times), kind="stable")
        fold_pairs = [(order[tr], order[va]) for tr, va in splitter.split(order)]
    elif resolved == "group":
        fold_pairs = list(splitter.split(X, y, groups))
    else:
        fold_pairs = list(splitter.split(X, y))

    folds = [(tr.tolist(), va.tolist()) for tr, va in fold_pairs]
    payload = {"fingerprint": fingerprint, "strategy": resolved, "n_rows": len(X), "folds": folds}
    from state.log import save_state
    save_state(folds_path, payload)
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


def validate_metric(task, metric):
    allowed = {"classification": {"roc_auc", "average_precision", "logloss", "accuracy", "balanced_accuracy", "f1"},
               "regression": {"rmse", "rmsle", "mae", "r2"}}
    if task not in allowed:
        raise ValueError(f"Unknown task '{task}'")
    if metric not in allowed[task]:
        raise ValueError(f"Metric '{metric}' is incompatible with {task}; choose {sorted(allowed[task])}")
    return metric

def cross_val_score(y_true, y_pred, task, metric=None):
    metric = validate_metric(task, metric or ("roc_auc" if task == "classification" else "r2"))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = np.isfinite(y_pred)
    if not valid.any():
        raise ValueError("No finite out-of-fold predictions are available for scoring")
    y_true, y_pred = y_true[valid], y_pred[valid]
    if metric == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    if metric == "average_precision":
        return average_precision_score(y_true, y_pred)
    if metric == "logloss":
        return log_loss(y_true, y_pred)
    if metric == "rmse":
        return mean_squared_error(y_true, y_pred) ** 0.5
    if metric == "rmsle":
        if np.any(y_true < 0):
            raise ValueError("RMSLE requires non-negative target values")
        return mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0))) ** 0.5
    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric == "r2":
        return r2_score(y_true, y_pred)
    if metric == "accuracy":
        return accuracy_score(y_true, (y_pred > 0.5).astype(int))
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, (y_pred > 0.5).astype(int))
    if metric == "f1":
        return f1_score(y_true, (y_pred > 0.5).astype(int))
    raise AssertionError(f"Unhandled metric: {metric}")


def metric_higher_is_better(metric):
    return metric in ("roc_auc", "average_precision", "r2", "accuracy", "balanced_accuracy", "f1")


def competition_diagnostics(X, y):
    """Return inspectable clues to guide metric, feature, and CV decisions."""
    notes = []
    date_names = [c for c in X if any(k in c.lower() for k in ("date", "time", "timestamp"))]
    if date_names:
        notes.append(f"Date/time-like columns need parsing and temporal-CV review: {date_names}")
    id_names = [c for c in X if c.lower() in {"id", "row_id", "sample_id", "customer_id", "user_id"}]
    if id_names:
        notes.append(f"Possible identifiers/group columns need review: {id_names}")
    numeric = X.select_dtypes(include=[np.number]).columns
    if len(numeric):
        missing = X[numeric].isna().mean().sort_values(ascending=False)
        high_missing = missing[missing >= 0.25]
        if len(high_missing):
            notes.append("Numeric columns with >=25% missing: " + ", ".join(
                f"{c} ({v:.0%})" for c, v in high_missing.items()))
    categorical = X.select_dtypes(exclude=[np.number]).columns
    high_card = [c for c in categorical if X[c].nunique(dropna=True) > max(50, len(X) * 0.1)]
    if high_card:
        notes.append(f"High-cardinality categoricals may benefit from leakage-safe encoding: {high_card}")
    if len(np.unique(y)) == 2:
        rate = float(np.mean(y))
        if min(rate, 1-rate) < 0.1:
            notes.append(f"Imbalanced target (minority share {min(rate, 1-rate):.1%}); verify metric and stratification.")
    if not notes:
        notes.append("No obvious structural clue found. Confirm the competition metric and split rules from its overview/data description.")
    return notes

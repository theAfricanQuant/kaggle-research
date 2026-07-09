import numpy as np
import pandas as pd


def target_encode(X_train, y, X_test, cat_cols, folds, smoothing=300):
    """Out-of-fold target encoding with smoothing toward the global mean.

    Each fold's encoding is computed only from the OTHER folds, so no row's
    encoded value ever depends on its own target — the classic target-encoding
    leak. Smoothing (m-estimate, m=smoothing) pulls low-count categories toward
    the global mean so a category seen twice doesn't get treated as a near-
    certain predictor. Test-set encoding uses statistics from the full
    training set, and unseen categories fall back to the global mean.
    """
    global_mean = float(np.mean(y))
    X_train = X_train.copy()
    X_test = X_test.copy() if X_test is not None else None

    for col in cat_cols:
        oof_col = np.full(len(X_train), global_mean)
        for tr_idx, va_idx in folds:
            fold_y = pd.Series(y[tr_idx])
            fold_cat = X_train[col].iloc[tr_idx].reset_index(drop=True)
            stats = fold_y.groupby(fold_cat.values).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            oof_col[va_idx] = X_train[col].iloc[va_idx].map(smoothed).fillna(global_mean).values
        X_train[f"te_{col}"] = oof_col

        full_stats = pd.Series(y).groupby(X_train[col].values).agg(["mean", "count"])
        full_smoothed = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
        if X_test is not None:
            X_test[f"te_{col}"] = X_test[col].astype(object).map(full_smoothed).fillna(global_mean).astype(float).values

    return X_train, X_test


def frequency_encode(X_train, X_test, cat_cols):
    """Adds a count-based frequency feature per categorical column, computed
    from train only (frequency is a property of the training distribution,
    not the label, so it doesn't need out-of-fold treatment).
    """
    X_train = X_train.copy()
    X_test = X_test.copy() if X_test is not None else None
    for col in cat_cols:
        freq = X_train[col].value_counts(normalize=True)
        X_train[f"freq_{col}"] = X_train[col].map(freq).fillna(0.0)
        if X_test is not None:
            X_test[f"freq_{col}"] = X_test[col].map(freq).fillna(0.0)
    return X_train, X_test


def numeric_interactions(X_train, X_test, max_cols=6):
    """Pairwise products between the top numeric columns (by variance).
    Kept intentionally small (max_cols choose 2) — GBDTs already model most
    interactions internally, so this targets the interactions a tree might
    need many splits to approximate.
    """
    X_train = X_train.copy()
    X_test = X_test.copy() if X_test is not None else None
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > max_cols:
        top = X_train[numeric_cols].var().sort_values(ascending=False).index[:max_cols]
    else:
        top = numeric_cols
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1, c2 = top[i], top[j]
            name = f"{c1}_x_{c2}"
            X_train[name] = X_train[c1] * X_train[c2]
            if X_test is not None:
                X_test[name] = X_test[c1] * X_test[c2]
    return X_train, X_test


def engineer_features(X_train, y, X_test, cat_cols, transforms, folds):
    if "target_encoding" in transforms and cat_cols:
        X_train, X_test = target_encode(X_train, y, X_test, cat_cols, folds)
    if "frequency" in transforms and cat_cols:
        X_train, X_test = frequency_encode(X_train, X_test, cat_cols)
    if "interactions" in transforms:
        X_train, X_test = numeric_interactions(X_train, X_test)
    return X_train, X_test

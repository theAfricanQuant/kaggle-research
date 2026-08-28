import numpy as np
import pandas as pd
import pytest

from pipeline.validate import (
    get_data,
    load_or_create_folds,
    resolve_cv_strategy,
    validate_metric,
)


def test_metric_must_match_task():
    assert validate_metric("classification", "roc_auc") == "roc_auc"
    assert validate_metric("regression", "rmse") == "rmse"
    with pytest.raises(ValueError, match="incompatible"):
        validate_metric("classification", "rmse")


def test_group_folds_never_split_an_entity_and_are_fingerprinted(tmp_path):
    X = pd.DataFrame({"x": np.arange(12)})
    y = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(4), 3)

    folds = load_or_create_folds(
        tmp_path, X, y, "classification", groups=groups,
        strategy="group", n_splits=4, fingerprint="run-a",
    )

    for train_idx, valid_idx in folds:
        assert set(groups[train_idx]).isdisjoint(groups[valid_idx])

    with pytest.raises(RuntimeError, match="do not match"):
        load_or_create_folds(
            tmp_path, X, y, "classification", groups=groups,
            strategy="group", n_splits=4, fingerprint="run-b",
        )


def test_time_folds_are_forward_chaining_even_when_input_is_unsorted(tmp_path):
    times = np.array([5, 1, 4, 2, 6, 3])
    X = pd.DataFrame({"x": np.arange(6)})
    y = np.arange(6, dtype=float)

    folds = load_or_create_folds(
        tmp_path, X, y, "regression", times=times,
        strategy="time", n_splits=2, fingerprint="time-run",
    )

    for train_idx, valid_idx in folds:
        assert times[train_idx].max() < times[valid_idx].min()


def test_auto_strategy_prefers_time_then_group():
    groups = np.array([1, 1, 2, 2])
    times = np.arange(4)
    assert resolve_cv_strategy("classification", groups=groups, times=times) == "time"
    assert resolve_cv_strategy("classification", groups=groups) == "group"
    assert resolve_cv_strategy("classification") == "stratified"
    assert resolve_cv_strategy("regression") == "kfold"


def test_data_loader_uses_submission_schema_and_extracts_split_columns(tmp_path):
    pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "entity": ["a", "a", "b", "b"],
        "date": [4, 1, 3, 2],
        "category": ["x", "y", "x", "z"],
        "SalePrice": [10.0, 11.0, 12.0, 13.0],
    }).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({
        "Id": [5, 6],
        "entity": ["c", "d"],
        "date": [5, 6],
        "category": ["x", "new"],
    }).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"Id": [5, 6], "SalePrice": [0.0, 0.0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )

    X, y, X_test, ids, cat_cols, split = get_data(
        tmp_path, group_col="entity", time_col="date"
    )

    assert list(y) == [11.0, 13.0, 12.0, 10.0]
    assert list(ids) == [5, 6]
    assert "entity" not in X and "date" not in X
    assert list(X_test.columns) == list(X.columns)
    assert cat_cols == ["category"]
    assert list(split["groups"]) == ["a", "b", "b", "a"]

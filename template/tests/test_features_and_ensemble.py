import numpy as np
import pandas as pd
import pytest

from pipeline.ensemble import apply_weights_to_test, cross_fitted_hill_climb, hill_climb
from pipeline.features import target_encode
from pipeline.validate import load_or_create_folds


def test_target_encoding_is_out_of_fold_for_unique_categories(tmp_path):
    X = pd.DataFrame({"cat": pd.Series(["a", "b", "c", "d"], dtype="category")})
    X_test = pd.DataFrame({"cat": pd.Series(["a", "new"], dtype="category")})
    y = np.array([0.0, 1.0, 0.0, 1.0])
    folds = load_or_create_folds(
        tmp_path, X, y, "regression", strategy="kfold",
        n_splits=2, random_state=42, fingerprint="te",
    )

    encoded, encoded_test = target_encode(X, y, X_test, ["cat"], folds, smoothing=10)

    assert np.all((encoded["te_cat"] > 0.4) & (encoded["te_cat"] < 0.6))
    assert encoded_test.loc[1, "te_cat"] == pytest.approx(y.mean())


def test_hill_climb_respects_minimizing_metrics_and_applies_weights():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    library = {
        "weak": np.array([0.0, 0.0, 0.0, 0.0]),
        "strong": np.array([0.0, 1.0, 2.0, 3.0]),
    }

    weights, blended, _ = hill_climb(y, library, "regression", "rmse")

    assert weights == {"strong": 1}
    assert np.allclose(blended, y)
    test_blend = apply_weights_to_test(weights, {
        "weak": np.array([1.0]), "strong": np.array([2.0])
    })
    assert np.allclose(test_blend, [2.0])


def test_cross_fitted_hill_climb_returns_held_out_meta_predictions():
    y = np.array([0, 1] * 10)
    strong = np.where(y == 1, 0.9, 0.1).astype(float)
    weak = np.linspace(0.2, 0.8, len(y))

    weights, cross_fitted, history = cross_fitted_hill_climb(
        y,
        {"strong": strong, "weak": weak},
        "classification",
        "roc_auc",
        n_splits=2,
    )

    assert weights == {"strong": 1}
    assert np.isfinite(cross_fitted).all()

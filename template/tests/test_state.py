import json

import numpy as np
import pytest

from state.experiments import load_experiment_library, save_experiment
from state.log import load_state, save_state
from state.run import StateMismatchError, fingerprint_data, initialize_or_validate_state


def run_config(data_fingerprint="abc"):
    return {
        "competition": "fixture",
        "data_fingerprint": data_fingerprint,
        "task": "classification",
        "metric": "roc_auc",
        "cv_strategy": "stratified",
        "group_col": None,
        "time_col": None,
        "n_splits": 5,
        "seed": 42,
        "optuna_trials": 10,
        "noise_seeds": 3,
    }


def test_state_initializes_and_resumes_without_resetting_progress():
    state = initialize_or_validate_state({}, run_config())
    state["iterations"].append({"iteration": 1})
    state["tried_hypotheses"].append("lgbm_defaults")
    state["accepted_feature_transforms"].append("frequency")
    state["next_iteration"] = 2

    resumed = initialize_or_validate_state(state, run_config())

    assert resumed["next_iteration"] == 2
    assert resumed["tried_hypotheses"] == ["lgbm_defaults"]
    assert resumed["accepted_feature_transforms"] == ["frequency"]


def test_state_rejects_changed_data_or_run_settings():
    state = initialize_or_validate_state({}, run_config())

    with pytest.raises(StateMismatchError, match="different data or run settings"):
        initialize_or_validate_state(state, run_config(data_fingerprint="changed"))


def test_state_rejects_legacy_unfingerprinted_state():
    with pytest.raises(StateMismatchError, match="incompatible schema"):
        initialize_or_validate_state({"iterations": []}, run_config())


def test_data_fingerprint_changes_with_input(tmp_path):
    (tmp_path / "train.csv").write_text("x,target\n1,0\n")
    first = fingerprint_data(tmp_path)
    (tmp_path / "train.csv").write_text("x,target\n2,1\n")

    assert fingerprint_data(tmp_path) != first


def test_state_write_and_experiment_metadata_are_persisted_atomically(tmp_path):
    state_path = tmp_path / "state" / "log.json"
    save_state(state_path, {"b": 2, "a": 1})
    assert load_state(state_path) == {"a": 1, "b": 2}

    exp_path = save_experiment(
        tmp_path / "state",
        "001_baseline",
        np.array([0.1, 0.9]),
        np.array([0.2]),
        0.75,
        metadata={"metric": "roc_auc", "best_params": {"depth": 3}},
    )
    metadata = json.loads((tmp_path / "state" / "experiments" / "001_baseline.json").read_text())
    oof, test, scores = load_experiment_library(tmp_path / "state")

    assert exp_path.endswith("001_baseline.npz")
    assert metadata["best_params"] == {"depth": 3}
    assert np.allclose(oof["001_baseline"], [0.1, 0.9])
    assert np.allclose(test["001_baseline"], [0.2])
    assert scores["001_baseline"] == pytest.approx(0.75)

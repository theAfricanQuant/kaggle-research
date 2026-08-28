from types import SimpleNamespace

import numpy as np
import pandas as pd

from main import _submit_current_best, check_cv_lb_alignment
from state.experiments import save_experiment


def test_submission_is_recorded_against_actual_experiment_and_not_duplicated(tmp_path):
    state_dir = tmp_path / "state"
    save_experiment(state_dir, "001_model", np.array([0.1, 0.9]), np.array([0.25]), 0.8)
    state = {
        "iterations": [{"iteration": 1, "cv_after": 0.8}],
        "submissions": [],
        "submitted_hashes": [],
    }
    args = SimpleNamespace(data_path=None, competition="fixture")
    calls = []

    def save_csv(ids, predictions, data_path, path):
        pd.DataFrame({"id": ids, "target": predictions}).to_csv(path, index=False)
        return path

    def submit(path, competition, message):
        calls.append((path, competition, message))
        return competition

    def poll(submission, competition):
        return 0.79

    _submit_current_best(
        state, state_dir, args, np.array([10]), "classification", tmp_path,
        submit, poll, save_csv, True,
    )
    _submit_current_best(
        state, state_dir, args, np.array([10]), "classification", tmp_path,
        submit, poll, save_csv, True,
    )

    assert len(calls) == 1
    assert state["submissions"][0]["experiment"] == "001_model"
    assert state["submissions"][0]["cv_score"] == 0.8
    assert state["submissions"][0]["lb_score"] == 0.79
    assert len(state["submitted_hashes"]) == 1


def test_alignment_uses_submission_records(caplog):
    caplog.set_level("INFO", logger="kaggle-research")
    state = {
        "submissions": [
            {"cv_score": score, "lb_score": score - 0.01}
            for score in [0.60, 0.65, 0.70, 0.75, 0.80]
        ]
    }

    check_cv_lb_alignment(state)

    assert "rank correlation" in caplog.text

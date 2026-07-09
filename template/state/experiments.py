import numpy as np
from pathlib import Path


def save_experiment(state_dir, name, oof, test_preds, cv_score):
    """Persists one hypothesis's OOF and test predictions to disk. Every
    experiment is saved, including ones that don't beat the running best —
    a weak-but-diverse model can still add value in hill climbing at the
    end, so nothing should be discarded during the loop itself.
    """
    exp_dir = Path(state_dir) / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / f"{name}.npz"
    np.savez(path, oof=oof, test_preds=test_preds if test_preds is not None else np.array([]),
              cv_score=cv_score)
    return str(path)


def load_experiment_library(state_dir):
    """Loads every saved experiment as (oof_library, test_library, scores)
    keyed by hypothesis name, for use by hill_climb + apply_weights_to_test.
    """
    exp_dir = Path(state_dir) / "experiments"
    oof_lib, test_lib, scores = {}, {}, {}
    if not exp_dir.exists():
        return oof_lib, test_lib, scores
    for path in exp_dir.glob("*.npz"):
        data = np.load(path)
        name = path.stem
        oof_lib[name] = data["oof"]
        if data["test_preds"].size:
            test_lib[name] = data["test_preds"]
        scores[name] = float(data["cv_score"])
    return oof_lib, test_lib, scores

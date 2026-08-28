import hashlib
import json
from pathlib import Path


STATE_SCHEMA_VERSION = 2


class StateMismatchError(RuntimeError):
    pass


def fingerprint_data(data_path):
    """Hashes the competition inputs that affect training and submission."""
    root = Path(data_path)
    digest = hashlib.sha256()
    found = False
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        path = root / name
        if not path.exists():
            continue
        found = True
        digest.update(name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"No competition CSV files found in {root}")
    return digest.hexdigest()


def fingerprint_config(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def initialize_or_validate_state(existing, run_config):
    """Creates state or proves existing state belongs to this exact run.

    Iteration/submission budgets are intentionally not part of run_config so a
    user can safely extend a run without invalidating its experiment library.
    """
    run_fingerprint = fingerprint_config(run_config)
    if not existing:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "run_config": run_config,
            "run_fingerprint": run_fingerprint,
            "iterations": [],
            "tried_hypotheses": [],
            "accepted_feature_transforms": [],
            "submissions": [],
            "submitted_hashes": [],
            "next_iteration": 1,
        }

    if existing.get("schema_version") != STATE_SCHEMA_VERSION:
        raise StateMismatchError(
            "Existing state uses an incompatible schema. Move or delete state/ "
            "before starting this version, or continue with the older code."
        )
    if existing.get("run_fingerprint") != run_fingerprint:
        raise StateMismatchError(
            "Existing state belongs to different data or run settings. Use a new "
            "project folder, or move/delete state/ before starting a clean run."
        )

    existing.setdefault("iterations", [])
    existing.setdefault("tried_hypotheses", [])
    existing.setdefault("accepted_feature_transforms", [])
    existing.setdefault("submissions", [])
    existing.setdefault("submitted_hashes", [])
    existing.setdefault(
        "next_iteration",
        max((entry.get("iteration", 0) for entry in existing["iterations"]), default=0) + 1,
    )
    return existing

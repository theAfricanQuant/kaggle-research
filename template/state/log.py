import json, os, tempfile
from pathlib import Path
from datetime import datetime


class LogEntry:
    def __init__(self, iteration, hypothesis, cv_before=None, cv_after=None,
                 delta=None, experiment_path=None,
                 lb_score=None, timestamp=None):
        self.iteration = iteration
        self.hypothesis = hypothesis
        self.cv_before = cv_before
        self.cv_after = cv_after
        self.delta = delta
        self.experiment_path = experiment_path
        self.lb_score = lb_score
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

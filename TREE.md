kaggle-research/
├── SKILL.md
├── README.md
└── template/
    ├── main.py
    ├── hardware.py
    ├── worker.py
    ├── bootstrap.sh
    ├── pyproject.toml
    ├── uv.lock
    ├── .python-version
    ├── kaggle_wrapper.ipynb
    ├── pipeline/
    │   ├── __init__.py
    │   ├── download.py
    │   ├── validate.py
    │   ├── train.py
    │   ├── tuner.py
    │   ├── features.py
    │   ├── ensemble.py
    │   └── submit.py
    ├── state/
    │   ├── log.py
    │   ├── run.py
    │   └── experiments.py
    └── tests/
        ├── test_state.py
        ├── test_validate.py
        ├── test_features_and_ensemble.py
        ├── test_submission.py
        └── test_router.py

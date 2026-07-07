import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


def train_lgbm(X, y, hw, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=500 if hw["gpu"] else 200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        models.append(model)
    return models, oof


def train_xgb(X, y, hw, n_splits=5):
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        model = xgb.XGBClassifier(
            n_estimators=500 if hw["gpu"] else 200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0,
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        models.append(model)
    return models, oof


def train_catboost(X, y, hw, n_splits=5):
    from catboost import CatBoostClassifier
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        model = CatBoostClassifier(
            iterations=500 if hw["gpu"] else 200,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
            task_type="GPU" if hw["gpu"] else "CPU",
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        models.append(model)
    return models, oof

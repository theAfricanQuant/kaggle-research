import numpy as np
import lightgbm as lgb
from pipeline.validate import get_splitter


def _get_model_cls(task):
    is_cls = task == "classification"
    return {
        "lgbm_cls": lgb.LGBMClassifier,
        "lgbm_reg": lgb.LGBMRegressor,
        "xgb_cls": None,
        "xgb_reg": None,
        "cat_cls": None,
        "cat_reg": None,
    }


def train_lgbm(X, y, hw, task, n_splits=5):
    is_cls = task == "classification"
    model_cls = lgb.LGBMClassifier if is_cls else lgb.LGBMRegressor
    skf = get_splitter(task, n_splits=n_splits)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        model = model_cls(
            n_estimators=500 if hw["gpu"] else 200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        model.fit(X[tr], y[tr])
        if is_cls:
            oof[va] = model.predict_proba(X[va])[:, 1]
        else:
            oof[va] = model.predict(X[va])
        models.append(model)
    return models, oof


def train_xgb(X, y, hw, task, n_splits=5):
    import xgboost as xgb
    is_cls = task == "classification"
    model_cls = xgb.XGBClassifier if is_cls else xgb.XGBRegressor
    skf = get_splitter(task, n_splits=n_splits)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        model = model_cls(
            n_estimators=500 if hw["gpu"] else 200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0,
        )
        model.fit(X[tr], y[tr])
        if is_cls:
            oof[va] = model.predict_proba(X[va])[:, 1]
        else:
            oof[va] = model.predict(X[va])
        models.append(model)
    return models, oof


def train_catboost(X, y, hw, task, n_splits=5):
    from catboost import CatBoostClassifier, CatBoostRegressor
    is_cls = task == "classification"
    model_cls = CatBoostClassifier if is_cls else CatBoostRegressor
    skf = get_splitter(task, n_splits=n_splits)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        model = model_cls(
            iterations=500 if hw["gpu"] else 200,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
            task_type="GPU" if hw["gpu"] else "CPU",
        )
        model.fit(X[tr], y[tr])
        if is_cls:
            oof[va] = model.predict_proba(X[va])[:, 1]
        else:
            oof[va] = model.predict(X[va])
        models.append(model)
    return models, oof

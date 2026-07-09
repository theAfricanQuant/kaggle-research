import numpy as np


def _predict(model, X, is_cls):
    return model.predict_proba(X)[:, 1] if is_cls else model.predict(X)


def train_lgbm(X, y, X_test, hw, task, folds, cat_cols=None, params=None, n_estimators=None):
    import lightgbm as lgb
    is_cls = task == "classification"
    model_cls = lgb.LGBMClassifier if is_cls else lgb.LGBMRegressor
    base_params = dict(
        n_estimators=n_estimators or (500 if hw["gpu"] else 200),
        learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1,
    )
    base_params.update(params or {})

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    best_iters = []
    for tr, va in folds:
        model = model_cls(**base_params)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                  categorical_feature=cat_cols or "auto",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = _predict(model, X.iloc[va], is_cls)
        best_iters.append(model.best_iteration_ or base_params["n_estimators"])
        if X_test is not None:
            test_preds += _predict(model, X_test, is_cls) / len(folds)
    return oof, test_preds, {"mean_best_iteration": int(np.mean(best_iters))}


def train_xgb(X, y, X_test, hw, task, folds, cat_cols=None, params=None, n_estimators=None):
    import xgboost as xgb
    is_cls = task == "classification"
    model_cls = xgb.XGBClassifier if is_cls else xgb.XGBRegressor
    base_params = dict(
        n_estimators=n_estimators or (500 if hw["gpu"] else 200),
        learning_rate=0.05, max_depth=6, random_state=42, verbosity=0,
        early_stopping_rounds=50, enable_categorical=bool(cat_cols),
    )
    base_params.update(params or {})
    if "n_estimators" not in (params or {}) and n_estimators:
        base_params["n_estimators"] = n_estimators

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    best_iters = []
    for tr, va in folds:
        model = model_cls(**base_params)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = _predict(model, X.iloc[va], is_cls)
        best_iters.append(getattr(model, "best_iteration", None) or base_params["n_estimators"])
        if X_test is not None:
            test_preds += _predict(model, X_test, is_cls) / len(folds)
    return oof, test_preds, {"mean_best_iteration": int(np.mean(best_iters))}


def train_depth1_xgb(X, y, X_test, hw, task, folds, cat_cols=None):
    return train_xgb(X, y, X_test, hw, task, folds, cat_cols=cat_cols, params={"max_depth": 1})


def train_catboost(X, y, X_test, hw, task, folds, cat_cols=None, params=None, n_estimators=None):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    is_cls = task == "classification"
    model_cls = CatBoostClassifier if is_cls else CatBoostRegressor
    base_params = dict(
        iterations=n_estimators or (500 if hw["gpu"] else 200),
        learning_rate=0.05, depth=6, random_seed=42, verbose=0,
        task_type="GPU" if hw["gpu"] else "CPU",
        early_stopping_rounds=50,
    )
    base_params.update(params or {})
    # Optuna's best_params only carries *suggested* keys, so the tuned refit
    # must re-add the static bootstrap_type that makes 'subsample' legal on GPU
    if "subsample" in base_params and "bootstrap_type" not in base_params:
        base_params["bootstrap_type"] = "Bernoulli"

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    best_iters = []
    for tr, va in folds:
        model = model_cls(**base_params)
        train_pool = Pool(X.iloc[tr], y[tr], cat_features=cat_cols or None)
        val_pool = Pool(X.iloc[va], y[va], cat_features=cat_cols or None)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        va_pool = Pool(X.iloc[va], cat_features=cat_cols or None)
        oof[va] = model.predict_proba(va_pool)[:, 1] if is_cls else model.predict(va_pool)
        best_iters.append(model.get_best_iteration() or base_params["iterations"])
        if X_test is not None:
            test_pool = Pool(X_test, cat_features=cat_cols or None)
            preds = model.predict_proba(test_pool)[:, 1] if is_cls else model.predict(test_pool)
            test_preds += preds / len(folds)
    return oof, test_preds, {"mean_best_iteration": int(np.mean(best_iters))}


def train_tuned(model_type, X, y, X_test, hw, task, folds, cat_cols, params, n_estimators):
    trainer = {"xgb": train_xgb, "lgbm": train_lgbm, "catboost": train_catboost}[model_type]
    return trainer(X, y, X_test, hw, task, folds, cat_cols=cat_cols, params=params, n_estimators=n_estimators)

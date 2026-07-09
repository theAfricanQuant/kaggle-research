import optuna
import numpy as np
from pipeline.validate import cross_val_score, metric_higher_is_better

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _metric_dir(metric):
    return "maximize" if metric_higher_is_better(metric) else "minimize"


def _xgb_objective(trial, X, y, task, hw, metric, folds, cat_cols):
    import xgboost as xgb
    is_cls = task == "classification"

    params = {
        "n_estimators": 2000, "random_state": 42, "verbosity": 0,
        "early_stopping_rounds": 50, "enable_categorical": bool(cat_cols),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
    }
    params["objective"] = "binary:logistic" if is_cls else "reg:squarederror"

    scores, best_iters = [], []
    for tr, va in folds:
        m = xgb.XGBClassifier(**params) if is_cls else xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        p = m.predict_proba(X.iloc[va])[:, 1] if is_cls else m.predict(X.iloc[va])
        scores.append(cross_val_score(y[va], p, task, metric))
        best_iters.append(getattr(m, "best_iteration", None) or params["n_estimators"])
    trial.set_user_attr("mean_best_iteration", int(np.mean(best_iters)))
    return np.mean(scores)


def _lgbm_objective(trial, X, y, task, hw, metric, folds, cat_cols):
    import lightgbm as lgb
    is_cls = task == "classification"

    params = {
        "n_estimators": 2000, "random_state": 42, "verbose": -1,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
    }
    params["objective"] = "binary" if is_cls else "regression"

    scores, best_iters = [], []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params) if is_cls else lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              categorical_feature=cat_cols or "auto",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict_proba(X.iloc[va])[:, 1] if is_cls else m.predict(X.iloc[va])
        scores.append(cross_val_score(y[va], p, task, metric))
        best_iters.append(m.best_iteration_ or params["n_estimators"])
    trial.set_user_attr("mean_best_iteration", int(np.mean(best_iters)))
    return np.mean(scores)


def _cat_objective(trial, X, y, task, hw, metric, folds, cat_cols):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    is_cls = task == "classification"

    params = {
        "iterations": 2000, "random_seed": 42, "verbose": 0,
        "task_type": "GPU" if hw["gpu"] else "CPU",
        "early_stopping_rounds": 50,
        "depth": trial.suggest_int("depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
    }

    scores, best_iters = [], []
    for tr, va in folds:
        m = CatBoostClassifier(**params) if is_cls else CatBoostRegressor(**params)
        train_pool = Pool(X.iloc[tr], y[tr], cat_features=cat_cols or None)
        val_pool = Pool(X.iloc[va], y[va], cat_features=cat_cols or None)
        m.fit(train_pool, eval_set=val_pool, verbose=0)
        va_pool = Pool(X.iloc[va], cat_features=cat_cols or None)
        p = m.predict_proba(va_pool)[:, 1] if is_cls else m.predict(va_pool)
        scores.append(cross_val_score(y[va], p, task, metric))
        best_iters.append(m.get_best_iteration() or params["iterations"])
    trial.set_user_attr("mean_best_iteration", int(np.mean(best_iters)))
    return np.mean(scores)


def _tune(objective_fn, X, y, task, hw, metric, folds, cat_cols, n_trials):
    study = optuna.create_study(direction=_metric_dir(metric), sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective_fn(t, X, y, task, hw, metric, folds, cat_cols), n_trials=n_trials)
    best_iter = study.best_trial.user_attrs.get("mean_best_iteration")
    return study.best_params, best_iter


def tune_xgb(X, y, task, hw, folds, cat_cols=None, metric="roc_auc", n_trials=50):
    return _tune(_xgb_objective, X, y, task, hw, metric, folds, cat_cols, n_trials)


def tune_lgbm(X, y, task, hw, folds, cat_cols=None, metric="roc_auc", n_trials=50):
    return _tune(_lgbm_objective, X, y, task, hw, metric, folds, cat_cols, n_trials)


def tune_catboost(X, y, task, hw, folds, cat_cols=None, metric="roc_auc", n_trials=50):
    return _tune(_cat_objective, X, y, task, hw, metric, folds, cat_cols, n_trials)

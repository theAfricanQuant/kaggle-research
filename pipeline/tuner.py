import optuna
import numpy as np
from pipeline.validate import get_splitter


def _metric_dir(metric):
    return "maximize" if metric in ("roc_auc", "r2", "accuracy", "f1") else "minimize"


def _metric_call(metric):
    return {
        "roc_auc": "roc_auc",
        "logloss": "logloss",
        "rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
        "accuracy": "accuracy",
        "f1": "f1",
    }.get(metric, "roc_auc")


def _xgb_objective(trial, X, y, task, hw, metric, n_splits):
    import xgboost as xgb
    is_cls = task == "classification"

    params = {
        "n_estimators": 2000,
        "random_state": 42,
        "verbosity": 0,
    }

    params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)

    params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    params["reg_alpha"] = trial.suggest_float("reg_alpha", 0, 10)
    params["reg_lambda"] = trial.suggest_float("reg_lambda", 0, 10)

    params["gamma"] = trial.suggest_float("gamma", 0, 10)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)

    if is_cls:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = metric if metric in ("logloss", "auc", "error") else "logloss"
    else:
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"

    skf = get_splitter(task, n_splits=n_splits)
    scores = []
    for tr, va in skf.split(X, y):
        m = xgb.XGBClassifier(**params) if is_cls else xgb.XGBRegressor(**params)
        m.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            verbose=False,
        )
        if is_cls:
            p = m.predict_proba(X[va])[:, 1]
            scores.append(optuna_metric(y[va], p, metric))
        else:
            p = m.predict(X[va])
            scores.append(optuna_metric(y[va], p, metric))
    return np.mean(scores)


def _lgbm_objective(trial, X, y, task, hw, metric, n_splits):
    import lightgbm as lgb
    is_cls = task == "classification"

    params = {
        "n_estimators": 2000,
        "random_state": 42,
        "verbose": -1,
    }

    params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
    params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)

    params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    params["reg_alpha"] = trial.suggest_float("reg_alpha", 0, 10)
    params["reg_lambda"] = trial.suggest_float("reg_lambda", 0, 10)

    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)

    if is_cls:
        params["objective"] = "binary"
        params["metric"] = metric if metric in ("auc", "binary_logloss") else "binary_logloss"
    else:
        params["objective"] = "regression"
        params["metric"] = "rmse"

    skf = get_splitter(task, n_splits=n_splits)
    scores = []
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**params) if is_cls else lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50)])
        if is_cls:
            p = m.predict_proba(X[va])[:, 1]
            scores.append(optuna_metric(y[va], p, metric))
        else:
            p = m.predict(X[va])
            scores.append(optuna_metric(y[va], p, metric))
    return np.mean(scores)


def _cat_objective(trial, X, y, task, hw, metric, n_splits):
    from catboost import CatBoostClassifier, CatBoostRegressor
    is_cls = task == "classification"

    params = {
        "iterations": 2000,
        "random_seed": 42,
        "verbose": 0,
        "task_type": "GPU" if hw["gpu"] else "CPU",
    }

    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)

    params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    params["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.5, 1.0)

    params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 0, 10)

    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)

    skf = get_splitter(task, n_splits=n_splits)
    scores = []
    for tr, va in skf.split(X, y):
        m = CatBoostClassifier(**params) if is_cls else CatBoostRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], early_stopping_rounds=50, verbose=0)
        if is_cls:
            p = m.predict_proba(X[va])[:, 1]
            scores.append(optuna_metric(y[va], p, metric))
        else:
            p = m.predict(X[va])
            scores.append(optuna_metric(y[va], p, metric))
    return np.mean(scores)


def optuna_metric(y_true, y_pred, metric):
    from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
    if metric == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    elif metric == "logloss":
        return log_loss(y_true, y_pred)
    elif metric == "rmse":
        return mean_squared_error(y_true, y_pred) ** 0.5
    elif metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "accuracy":
        return accuracy_score(y_true, (y_pred > 0.5).astype(int)) if y_pred.ndim == 1 else accuracy_score(y_true, y_pred)
    elif metric == "f1":
        return f1_score(y_true, (y_pred > 0.5).astype(int))
    return roc_auc_score(y_true, y_pred)


def tune_xgb(X, y, task, hw, metric="roc_auc", n_trials=50, n_splits=5):
    study = optuna.create_study(direction=_metric_dir(metric), sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _xgb_objective(t, X, y, task, hw, metric, n_splits), n_trials=n_trials)
    return study.best_params


def tune_lgbm(X, y, task, hw, metric="roc_auc", n_trials=50, n_splits=5):
    study = optuna.create_study(direction=_metric_dir(metric), sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _lgbm_objective(t, X, y, task, hw, metric, n_splits), n_trials=n_trials)
    return study.best_params


def tune_catboost(X, y, task, hw, metric="roc_auc", n_trials=50, n_splits=5):
    study = optuna.create_study(direction=_metric_dir(metric), sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _cat_objective(t, X, y, task, hw, metric, n_splits), n_trials=n_trials)
    return study.best_params

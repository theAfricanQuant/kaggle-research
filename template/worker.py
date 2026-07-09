import logging
import numpy as np

log = logging.getLogger("kaggle-research")

# The single source of truth for hypothesis names. main.py validates its
# routing lists against this at startup — a mismatch between router and
# worker names once made the whole loop silently no-op, so the check is
# structural now, not a convention.
KNOWN_HYPOTHESES = frozenset({
    "lgbm_defaults", "fe_target_encoding", "fe_frequency", "fe_interactions",
    "xgb_defaults", "catboost_defaults", "depth1_xgb_ensemble",
    "optuna_xgb", "optuna_lgbm", "optuna_catboost",
})


def run_hypothesis(hyp, ctx):
    """Dispatches a hypothesis name to its implementation. ctx carries
    everything workers need: X, y, X_test, cat_cols, folds, hw, task,
    metric, feature_state (the compounding feature set from prior
    iterations), optuna_trials.

    Every worker returns {"hypothesis", "cv_score", "oof", "test_preds"}.
    oof/test_preds are always populated (never None) so every experiment
    can be persisted and later included in hill climbing — a hypothesis
    that doesn't beat the running best is still valuable ensemble material.
    """
    handlers = {
        "lgbm_defaults": _baseline,
        "fe_target_encoding": _with_feature_engineering,
        "fe_frequency": _with_feature_engineering,
        "fe_interactions": _with_feature_engineering,
        "xgb_defaults": _xgb_baseline,
        "catboost_defaults": _catboost_baseline,
        "optuna_xgb": _tuned_model,
        "optuna_lgbm": _tuned_model,
        "optuna_catboost": _tuned_model,
        "depth1_xgb_ensemble": _depth1_ensemble,
    }
    handler = handlers.get(hyp)
    if handler is None:
        log.warning(f"Unknown hypothesis '{hyp}' — skipping")
        return None
    return handler(hyp, ctx)


def _baseline(hyp, ctx):
    from pipeline.train import train_lgbm
    from pipeline.validate import cross_val_score

    X, X_test = ctx["feature_state"]["X"], ctx["feature_state"]["X_test"]
    oof, test_preds, _ = train_lgbm(X, ctx["y"], X_test, ctx["hw"], ctx["task"], ctx["folds"], ctx["cat_cols"])
    cv = cross_val_score(ctx["y"], oof, ctx["task"], ctx["metric"])
    return {"hypothesis": "lgbm_defaults", "cv_score": cv, "oof": oof, "test_preds": test_preds}


def _xgb_baseline(hyp, ctx):
    from pipeline.train import train_xgb
    from pipeline.validate import cross_val_score

    X, X_test = ctx["feature_state"]["X"], ctx["feature_state"]["X_test"]
    oof, test_preds, _ = train_xgb(X, ctx["y"], X_test, ctx["hw"], ctx["task"], ctx["folds"], ctx["cat_cols"])
    cv = cross_val_score(ctx["y"], oof, ctx["task"], ctx["metric"])
    return {"hypothesis": "xgb_defaults", "cv_score": cv, "oof": oof, "test_preds": test_preds}


def _catboost_baseline(hyp, ctx):
    from pipeline.train import train_catboost
    from pipeline.validate import cross_val_score

    X, X_test = ctx["feature_state"]["X"], ctx["feature_state"]["X_test"]
    oof, test_preds, _ = train_catboost(X, ctx["y"], X_test, ctx["hw"], ctx["task"], ctx["folds"], ctx["cat_cols"])
    cv = cross_val_score(ctx["y"], oof, ctx["task"], ctx["metric"])
    return {"hypothesis": "catboost_defaults", "cv_score": cv, "oof": oof, "test_preds": test_preds}


def _with_feature_engineering(hyp, ctx):
    """Feature-engineering hypotheses mutate ctx["feature_state"] in place
    when they improve CV (decided by the caller in main.py) so later
    hypotheses build on the winning feature set instead of starting over
    from raw data every iteration.
    """
    from pipeline.features import engineer_features
    from pipeline.train import train_lgbm
    from pipeline.validate import cross_val_score

    transform = {"fe_target_encoding": "target_encoding", "fe_frequency": "frequency",
                 "fe_interactions": "interactions"}[hyp]
    base_X, base_X_test = ctx["feature_state"]["X"], ctx["feature_state"]["X_test"]
    X_fe, X_test_fe = engineer_features(base_X, ctx["y"], base_X_test, ctx["cat_cols"], [transform], ctx["folds"])

    oof, test_preds, _ = train_lgbm(X_fe, ctx["y"], X_test_fe, ctx["hw"], ctx["task"], ctx["folds"], ctx["cat_cols"])
    cv = cross_val_score(ctx["y"], oof, ctx["task"], ctx["metric"])
    return {"hypothesis": hyp, "cv_score": cv, "oof": oof, "test_preds": test_preds,
            "candidate_features": (X_fe, X_test_fe)}


def _tuned_model(hyp, ctx):
    from pipeline.tuner import tune_xgb, tune_lgbm, tune_catboost
    from pipeline.train import train_tuned
    from pipeline.validate import cross_val_score

    model_type = hyp.split("_", 1)[1]
    tuner = {"xgb": tune_xgb, "lgbm": tune_lgbm, "catboost": tune_catboost}[model_type]

    X, X_test = ctx["feature_state"]["X"], ctx["feature_state"]["X_test"]
    log.info(f"  Optuna tuning {model_type} ({ctx['optuna_trials']} trials, metric={ctx['metric']})...")
    best_params, best_iter = tuner(X, ctx["y"], ctx["task"], ctx["hw"], ctx["folds"],
                                    cat_cols=ctx["cat_cols"], metric=ctx["metric"], n_trials=ctx["optuna_trials"])
    log.info(f"  Best {model_type} params: {best_params} (mean_best_iteration={best_iter})")

    oof, test_preds, _ = train_tuned(model_type, X, ctx["y"], X_test, ctx["hw"], ctx["task"],
                                      ctx["folds"], ctx["cat_cols"], best_params, best_iter)
    cv = cross_val_score(ctx["y"], oof, ctx["task"], ctx["metric"])
    return {"hypothesis": hyp, "cv_score": cv, "oof": oof, "test_preds": test_preds}


def _depth1_ensemble(hyp, ctx):
    from pipeline.train import train_depth1_xgb, train_lgbm
    from pipeline.validate import cross_val_score

    X, X_test = ctx["feature_state"]["X"], ctx["feature_state"]["X_test"]
    oof_d1, test_d1, _ = train_depth1_xgb(X, ctx["y"], X_test, ctx["hw"], ctx["task"], ctx["folds"], ctx["cat_cols"])
    oof_l, test_l, _ = train_lgbm(X, ctx["y"], X_test, ctx["hw"], ctx["task"], ctx["folds"], ctx["cat_cols"])
    oof = np.column_stack([oof_d1, oof_l]).mean(axis=1)
    test_preds = np.column_stack([test_d1, test_l]).mean(axis=1) if X_test is not None else None
    cv = cross_val_score(ctx["y"], oof, ctx["task"], ctx["metric"])
    return {"hypothesis": "depth1_xgb_ensemble", "cv_score": cv, "oof": oof, "test_preds": test_preds}

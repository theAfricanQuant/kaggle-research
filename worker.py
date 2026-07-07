def dispatch_workers(hypotheses, data_path, hw):
    results = []
    for hyp in hypotheses:
        res = _run_hypothesis(hyp, data_path, hw)
        if res:
            results.append(res)
    return results


def _run_hypothesis(hyp, data_path, hw):
    from pipeline.features import engineer_features
    from pipeline.train import train_model
    from pipeline.validate import cross_val_score

    if hyp == "stratified_5fold_lgbm_defaults":
        return _baseline(data_path, hw)
    elif hyp == "feature_engineering_target_encoding":
        return _with_feature_engineering(data_path, hw, ["target_encoding"])
    elif hyp == "feature_engineering_interactions":
        return _with_feature_engineering(data_path, hw, ["interactions"])
    elif hyp == "try_xgboost_with_tuning":
        return _train_specific(data_path, hw, "xgb")
    elif hyp == "try_catboost_with_tuning":
        return _train_specific(data_path, hw, "catboost")
    elif hyp == "average_lgbm_xgb_catboost":
        return _ensemble_average(data_path, hw)
    elif hyp == "blend_with_meta_model":
        return _blend(data_path, hw)
    elif hyp == "stack_ensemble_all_top_models":
        return _stack(data_path, hw)
    else:
        return None


def _baseline(data_path, hw):
    import pandas as pd
    from pipeline.train import train_lgbm
    from pipeline.validate import cross_val_score, get_data

    X, y = get_data(data_path)
    model, oof = train_lgbm(X, y, hw)
    cv = cross_val_score(y, oof)
    return {"hypothesis": "stratified_5fold_lgbm_defaults", "cv_score": cv,
            "model_path": None, "preds_path": None}


def _with_feature_engineering(data_path, hw, transforms):
    import pandas as pd
    from pipeline.features import engineer_features
    from pipeline.train import train_lgbm
    from pipeline.validate import cross_val_score, get_data

    X, y = get_data(data_path)
    X_fe = engineer_features(X, y, transforms)
    model, oof = train_lgbm(X_fe, y, hw)
    cv = cross_val_score(y, oof)
    return {"hypothesis": f"fe_{'_'.join(transforms)}", "cv_score": cv,
            "model_path": None, "preds_path": None}


def _train_specific(data_path, hw, model_type):
    from pipeline.validate import get_data
    from pipeline.train import train_lgbm, train_xgb, train_catboost
    from pipeline.validate import cross_val_score

    X, y = get_data(data_path)
    fn = {"xgb": train_xgb, "catboost": train_catboost}.get(model_type)
    if not fn:
        return None
    model, oof = fn(X, y, hw)
    cv = cross_val_score(y, oof)
    return {"hypothesis": f"try_{model_type}", "cv_score": cv,
            "model_path": None, "preds_path": None}


def _ensemble_average(data_path, hw):
    from pipeline.validate import get_data
    from pipeline.train import train_lgbm, train_xgb, train_catboost
    from pipeline.validate import cross_val_score
    import numpy as np

    X, y = get_data(data_path)
    _, oof_l = train_lgbm(X, y, hw)
    _, oof_x = train_xgb(X, y, hw)
    _, oof_c = train_catboost(X, y, hw)
    oof_avg = np.column_stack([oof_l, oof_x, oof_c]).mean(axis=1)
    cv = cross_val_score(y, oof_avg)
    return {"hypothesis": "average_lgbm_xgb_catboost", "cv_score": cv,
            "model_path": None, "preds_path": None}


def _blend(data_path, hw):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from pipeline.validate import get_data
    from pipeline.train import train_lgbm, train_xgb, train_catboost

    X, y = get_data(data_path)
    X_tr, X_bl, y_tr, y_bl = train_test_split(X, y, test_size=0.2, random_state=42)

    def oof_on_holdout(trainer, Xtr, ytr, Xbl):
        m, _ = trainer(Xtr, ytr, hw)
        return m.predict_proba(Xbl)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xbl)

    bl_l = oof_on_holdout(train_lgbm, X_tr, y_tr, X_bl)
    bl_x = oof_on_holdout(train_xgb, X_tr, y_tr, X_bl)
    bl_c = oof_on_holdout(train_catboost, X_tr, y_tr, X_bl)
    meta = np.column_stack([bl_l, bl_x, bl_c])
    blender = LogisticRegression(max_iter=1000).fit(meta, y_bl)
    cv = blender.score(meta, y_bl)
    return {"hypothesis": "blend_with_meta_model", "cv_score": cv,
            "model_path": None, "preds_path": None}


def _stack(data_path, hw):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from pipeline.validate import get_data
    from pipeline.train import train_lgbm, train_xgb, train_catboost
    from pipeline.validate import cross_val_score

    X, y = get_data(data_path)
    _, oof_l = train_lgbm(X, y, hw)
    _, oof_x = train_xgb(X, y, hw)
    _, oof_c = train_catboost(X, y, hw)
    meta = np.column_stack([oof_l, oof_x, oof_c])
    stacker = LogisticRegression(max_iter=1000).fit(meta, y)
    oof_stack = stacker.predict(meta)
    cv = cross_val_score(y, oof_stack)
    return {"hypothesis": "stack_ensemble_all_top_models", "cv_score": cv,
            "model_path": None, "preds_path": None}

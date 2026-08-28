import numpy as np
from scipy.stats import rankdata


def average_ensemble(preds_list, weights=None):
    preds = np.column_stack(preds_list)
    if weights:
        return (preds * weights).sum(axis=1) / sum(weights)
    return preds.mean(axis=1)


def rank_average(preds_list):
    """Averages ranks instead of raw scores. Use for AUC/ranking metrics
    when ensemble members live on different scales (e.g. a neural net's
    logits vs a GBDT's calibrated probabilities) — comparing raw values
    would let the more confidently-scaled model dominate the average.
    """
    ranks = [rankdata(p) / len(p) for p in preds_list]
    return np.mean(ranks, axis=0)


def hill_climb(y_true, oof_library, task, metric, max_rounds=100, tol=1e-5):
    """Caruana-style ensemble selection: greedily add the library member
    (with replacement) that most improves the blended OOF score, stopping
    when no addition helps. With-replacement selection lets a strong model
    get more "votes" than a weak one without hand-picking weights, and
    naturally down-weights members that don't help by simply not
    re-selecting them.

    oof_library: dict of {name: oof_predictions}. Returns (weights_dict,
    final_blended_oof, history) where weights_dict counts how many times
    each member was selected.
    """
    from pipeline.validate import cross_val_score, metric_higher_is_better
    higher_is_better = metric_higher_is_better(metric)

    names = list(oof_library.keys())
    preds = {n: np.asarray(oof_library[n]) for n in names}

    # seed with the single best member
    scores = {n: cross_val_score(y_true, preds[n], task, metric) for n in names}
    best_name = max(scores, key=scores.get) if higher_is_better else min(scores, key=scores.get)
    selected = [best_name]
    current_blend = preds[best_name].copy()
    current_score = scores[best_name]
    history = [(best_name, current_score)]

    for _ in range(max_rounds - 1):
        candidate_scores = {}
        for n in names:
            trial_blend = (current_blend * len(selected) + preds[n]) / (len(selected) + 1)
            candidate_scores[n] = cross_val_score(y_true, trial_blend, task, metric)

        best_candidate = max(candidate_scores, key=candidate_scores.get) if higher_is_better \
            else min(candidate_scores, key=candidate_scores.get)
        best_candidate_score = candidate_scores[best_candidate]

        improved = (best_candidate_score > current_score + tol) if higher_is_better \
            else (best_candidate_score < current_score - tol)
        if not improved:
            break

        selected.append(best_candidate)
        current_blend = (current_blend * (len(selected) - 1) + preds[best_candidate]) / len(selected)
        current_score = best_candidate_score
        history.append((best_candidate, current_score))

    weights = {n: selected.count(n) for n in set(selected)}
    return weights, current_blend, history


def cross_fitted_hill_climb(y_true, oof_library, task, metric, n_splits=5, random_state=42):
    """Selects ensemble members inside meta-folds and scores on held-out rows.

    The final weights are learned from all finite OOF rows for test inference,
    while returned OOF predictions are cross-fitted and therefore suitable for
    an honest estimate of ensemble-selection performance.
    """
    from pipeline.validate import get_splitter

    names = list(oof_library)
    if len(names) < 2:
        raise ValueError("Cross-fitted ensemble selection requires at least two experiments")

    predictions = {name: np.asarray(oof_library[name]) for name in names}
    finite = np.logical_and.reduce([np.isfinite(predictions[name]) for name in names])
    valid_indices = np.flatnonzero(finite)
    if len(valid_indices) < 4:
        raise ValueError("Too few finite OOF rows for cross-fitted ensemble selection")

    y_true = np.asarray(y_true)
    y_valid = y_true[valid_indices]
    max_splits = len(valid_indices)
    if task == "classification":
        _, counts = np.unique(y_valid, return_counts=True)
        max_splits = min(max_splits, int(counts.min()))
    n_splits = min(n_splits, max_splits)
    if n_splits < 2:
        raise ValueError("Too few rows per class for cross-fitted ensemble selection")

    strategy = "stratified" if task == "classification" else "kfold"
    splitter = get_splitter(task, strategy=strategy, n_splits=n_splits, random_state=random_state)
    positions = np.arange(len(valid_indices))
    split_iter = splitter.split(positions, y_valid)
    cross_fitted = np.full(len(y_true), np.nan)

    for train_pos, valid_pos in split_iter:
        train_idx = valid_indices[train_pos]
        holdout_idx = valid_indices[valid_pos]
        train_library = {name: predictions[name][train_idx] for name in names}
        weights, _, _ = hill_climb(y_true[train_idx], train_library, task, metric)
        holdout_library = {name: predictions[name][holdout_idx] for name in names}
        cross_fitted[holdout_idx] = apply_weights_to_test(weights, holdout_library)

    full_library = {name: predictions[name][valid_indices] for name in names}
    final_weights, _, history = hill_climb(y_valid, full_library, task, metric)
    return final_weights, cross_fitted, history


def apply_weights_to_test(weights, test_pred_library):
    """Applies hill-climbed weights (member -> selection count) to the
    corresponding test-set prediction library to produce the final
    submission-ready prediction.
    """
    total = sum(weights.values())
    blend = np.zeros(len(next(iter(test_pred_library.values()))))
    for name, count in weights.items():
        blend += test_pred_library[name] * count
    return blend / total

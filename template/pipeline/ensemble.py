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

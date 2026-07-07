import numpy as np


def average_ensemble(preds_list, weights=None):
    preds = np.column_stack(preds_list)
    if weights:
        return (preds * weights).sum(axis=1) / sum(weights)
    return preds.mean(axis=1)


def weighted_by_cv(preds_list, cv_scores):
    return average_ensemble(preds_list, weights=cv_scores)


def weighted_by_inv_covariance(preds_list):
    from numpy import corrcoef
    preds = np.column_stack(preds_list)
    corr = corrcoef(preds.T)
    np.fill_diagonal(corr, 0.0)
    w = 1 / np.maximum(np.mean(corr, axis=1), 1e-8)
    return average_ensemble(preds_list, weights=w.tolist())

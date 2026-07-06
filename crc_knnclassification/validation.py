"""Cross-validation, hyperparameter tuning, and medoid metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .knn import TunableWeightedKNN, WeightedKNN, lp_distance_matrix


def confidence_interval_90(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean_value = float(np.mean(values))

    if n < 2:
        return mean_value, mean_value, mean_value, 0.0

    sigma = float(np.std(values, ddof=1))
    margin = 1.645 * sigma / np.sqrt(n)
    return mean_value, float(mean_value - margin), float(mean_value + margin), sigma


def cross_validate_tunable_knn(
    X,
    y,
    k=5,
    p=1.5,
    weighting="gaussian",
    n_splits=5,
    random_state=42,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_f1 = []
    fold_accuracy = []

    for train_idx, test_idx in skf.split(X, y):
        model = TunableWeightedKNN(k=k, p=p, weighting=weighting)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        fold_accuracy.append(np.mean(y_pred == y[test_idx]))
        fold_f1.append(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0))

    acc_mean, acc_low, acc_high, acc_sigma = confidence_interval_90(fold_accuracy)
    f1_mean, f1_low, f1_high, f1_sigma = confidence_interval_90(fold_f1)

    return {
        "k": k,
        "p": p,
        "weighting": weighting,
        "fold_accuracy": fold_accuracy,
        "fold_f1": fold_f1,
        "Accuracy_CV": acc_mean,
        "Accuracy_CI90_low": acc_low,
        "Accuracy_CI90_high": acc_high,
        "Accuracy_sigma": acc_sigma,
        "F1_CV": f1_mean,
        "F1_CI90_low": f1_low,
        "F1_CI90_high": f1_high,
        "F1_sigma": f1_sigma,
    }


def run_knn_hyperparameter_tuning(
    X,
    y,
    k_values=None,
    p_values=None,
    weighting_functions=None,
    n_splits=5,
    random_state=42,
):
    k_values = list(range(3, 16, 2)) if k_values is None else k_values
    p_values = [1.0, 1.5, 2.0] if p_values is None else p_values
    weighting_functions = (
        ["uniform", "inverse-distance", "gaussian"]
        if weighting_functions is None
        else weighting_functions
    )

    rows = []
    for k in k_values:
        for p in p_values:
            for weighting in weighting_functions:
                rows.append(
                    cross_validate_tunable_knn(
                        X=X,
                        y=y,
                        k=k,
                        p=p,
                        weighting=weighting,
                        n_splits=n_splits,
                        random_state=random_state,
                    )
                )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["F1_CV", "Accuracy_CV"], ascending=[False, False])
        .reset_index(drop=True)
    )


def class_medoid(X_class, p=1.5):
    X_class = np.asarray(X_class, dtype=float)

    if len(X_class) == 0:
        raise ValueError("Cannot compute medoid for an empty class.")
    if len(X_class) == 1:
        return X_class[0]

    distances = lp_distance_matrix(X_class, X_class, p=p)
    return X_class[np.argmin(np.sum(distances, axis=1))]


def compute_class_medoids(X, y, p=1.5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    return {c: class_medoid(X[y == c], p=p) for c in np.unique(y)}


def mean_distance_of_medoid(X, y, medoids, p=1.5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    distances = []

    for xi, yi in zip(X, y):
        medoid = medoids[yi]
        distances.append(lp_distance_matrix(np.array([xi]), np.array([medoid]), p=p)[0, 0])

    return float(np.mean(distances))


def cross_validate_knn(X, y, k=5, p=1.5, n_splits=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    fold_f1 = []
    fold_mdm = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = WeightedKNN(k=k, p=p).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_scores.append(np.mean(y_pred == y_test))
        fold_f1.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        medoids = compute_class_medoids(X_train, y_train, p=p)
        valid_mask = np.array([yi in medoids for yi in y_test])
        fold_mdm.append(mean_distance_of_medoid(X_test[valid_mask], y_test[valid_mask], medoids, p=p))

    score_mean, score_ci_low, score_ci_high, score_sigma = confidence_interval_90(fold_scores)
    f1_mean, f1_ci_low, f1_ci_high, f1_sigma = confidence_interval_90(fold_f1)
    mdm_mean, mdm_ci_low, mdm_ci_high, mdm_sigma = confidence_interval_90(fold_mdm)

    return {
        "fold_scores": np.array(fold_scores),
        "fold_f1": np.array(fold_f1),
        "fold_mdm": np.array(fold_mdm),
        "Score_CV": score_mean,
        "F1_CV": f1_mean,
        "MDM_CV": mdm_mean,
        "Score_CI90": (score_ci_low, score_ci_high),
        "F1_CI90": (f1_ci_low, f1_ci_high),
        "MDM_CI90": (mdm_ci_low, mdm_ci_high),
        "Score_sigma": score_sigma,
        "F1_sigma": f1_sigma,
        "MDM_sigma": mdm_sigma,
    }


def interpret_classifier_performance(f1_cv, mdm_cv, ci90_width):
    return {
        "Prediction": "Pass" if f1_cv > 0.75 else "Fail",
        "Cohesion": "Pass" if mdm_cv < 0.2 else "Fail",
        "Stability": "Pass" if ci90_width < 0.15 else "Fail",
    }

"""Class labelling utilities and label-structure evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import LabelEncoder


def assign_act_classes(duration, pressure):
    duration = np.asarray(duration, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    labels = []

    for d, p in zip(duration, pressure):
        if (p <= 1.0) and (d <= 10.0):
            labels.append("ACT-1")
        elif (p <= 1.0) and (d > 10.0):
            labels.append("ACT-2")
        elif (p > 1.0) and (p < 3.0) and (d <= 10.0):
            labels.append("ACT-3")
        elif (p >= 3.0) and (d <= 10.0):
            labels.append("ACT-4")
        else:
            labels.append("Unclassified")

    return np.asarray(labels)


def make_labels_for_k_classes(duration, pressure, k):
    duration = np.asarray(duration, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    labels = []

    for d, p in zip(duration, pressure):
        if k == 2:
            labels.append("Atmospheric" if p <= 1.0 else "Pressurised")
        elif k == 3:
            if p <= 1.0:
                labels.append("Atmospheric")
            elif 1.0 < p < 3.0:
                labels.append("Medium-pressure")
            else:
                labels.append("High-pressure")
        elif k == 4:
            if (p <= 1.0) and (d <= 10.0):
                labels.append("ACT-1")
            elif (p <= 1.0) and (d > 10.0):
                labels.append("ACT-2")
            elif (1.0 < p < 3.0) and (d <= 10.0):
                labels.append("ACT-3")
            elif (p >= 3.0) and (d <= 10.0):
                labels.append("ACT-4")
            else:
                labels.append("Unclassified")
        elif k == 5:
            if (p <= 1.0) and (d <= 10.0):
                labels.append("Atmospheric-short")
            elif (p <= 1.0) and (d > 10.0):
                labels.append("Atmospheric-long")
            elif (1.0 < p < 3.0) and (d <= 10.0):
                labels.append("Medium-short")
            elif (1.0 < p < 3.0) and (d > 10.0):
                labels.append("Medium-long")
            elif (p >= 3.0) and (d <= 10.0):
                labels.append("High-short")
            else:
                labels.append("Unclassified")
        elif k == 6:
            if (p <= 1.0) and (d <= 10.0):
                labels.append("Atmospheric-short")
            elif (p <= 1.0) and (d > 10.0):
                labels.append("Atmospheric-long")
            elif (1.0 < p < 3.0) and (d <= 10.0):
                labels.append("Medium-short")
            elif (1.0 < p < 3.0) and (d > 10.0):
                labels.append("Medium-long")
            elif (p >= 3.0) and (d <= 10.0):
                labels.append("High-short")
            elif (p >= 3.0) and (d > 10.0):
                labels.append("High-long")
            else:
                labels.append("Unclassified")
        else:
            raise ValueError("k must be between 2 and 6.")

    return np.asarray(labels)


def evaluate_label_structure(X, labels):
    labels = np.asarray(labels)
    valid_mask = labels != "Unclassified"
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(np.unique(labels_valid)) < 2:
        raise ValueError("At least two classes are required.")

    encoded = LabelEncoder().fit_transform(labels_valid)
    return (
        silhouette_score(X_valid, encoded),
        davies_bouldin_score(X_valid, encoded),
        calinski_harabasz_score(X_valid, encoded),
        X_valid,
        labels_valid,
    )


def evaluate_all_class_structures(duration, pressure):
    X = np.column_stack([pressure, duration])
    structure_names = {
        2: "Atmospheric-pressure; Pressurised",
        3: "Atmospheric-pressure; Medium-pressure; High-pressure",
        4: "Atmospheric-short; Atmospheric-long; Medium-short; High-short",
        5: "Atmospheric-short; Atmospheric-long; Medium-short; Medium-long; High-short",
        6: "Full pressure and duration combination",
    }
    rows = []

    for k in [2, 3, 4, 5, 6]:
        labels = make_labels_for_k_classes(duration, pressure, k)
        silhouette, dbi, chi, _, _ = evaluate_label_structure(X, labels)
        rows.append(
            {
                "ACT Classes": k,
                "Structure assumption": structure_names[k],
                "Silhouette": silhouette,
                "DBI": dbi,
                "CHI": chi,
            }
        )

    return pd.DataFrame(rows)

"""K-means validation against ACT labels."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from .labeling import assign_act_classes


def match_kmeans_clusters_to_act(y_true, y_cluster):
    cm = confusion_matrix(y_true, y_cluster)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    return np.array([mapping[c] for c in y_cluster])


def validate_kmeans_against_act(duration, pressure):
    X = np.column_stack([pressure, duration])
    act_labels = assign_act_classes(duration, pressure)
    valid_mask = act_labels != "Unclassified"
    X_valid = X[valid_mask]
    act_labels_valid = act_labels[valid_mask]

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(act_labels_valid)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    y_cluster = kmeans.fit_predict(X_valid)
    y_cluster_mapped = match_kmeans_clusters_to_act(y_true, y_cluster)

    rows = []
    for class_id in np.unique(y_true):
        mask = y_true == class_id
        rows.append(
            {
                "Labeled ACT": encoder.inverse_transform([class_id])[0],
                "K-means cluster": f"Cluster-{class_id + 1}",
                "Agreement (%)": 100.0 * np.mean(y_true[mask] == y_cluster_mapped[mask]),
            }
        )

    validation_metrics = pd.DataFrame(
        {
            "Metric": ["Overall agreement (%)", "ARI", "NMI"],
            "Value": [
                100.0 * np.mean(y_true == y_cluster_mapped),
                adjusted_rand_score(y_true, y_cluster),
                normalized_mutual_info_score(y_true, y_cluster),
            ],
        }
    )
    return pd.DataFrame(rows), validation_metrics

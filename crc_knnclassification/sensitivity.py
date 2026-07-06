"""Global sensitivity analysis using Spearman rank correlation."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def global_sensitivity_srcc(X, y, model):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    n_features = X.shape[1]
    confidences = model.predict_posterior_confidence(X)
    records = []

    for c in classes:
        class_mask = y == c
        Xc = X[class_mask]
        Cc = confidences[class_mask]

        for m in range(n_features):
            feature_values = Xc[:, m]

            if len(np.unique(feature_values)) < 2 or len(np.unique(Cc)) < 2:
                rho = np.nan
                pval = np.nan
            else:
                rho, pval = spearmanr(feature_values, Cc)

            abs_rho = np.abs(rho) if not np.isnan(rho) else np.nan

            if np.isnan(abs_rho):
                level = "Undefined"
            elif abs_rho >= 0.7:
                level = "High"
            elif abs_rho >= 0.4:
                level = "Moderate"
            else:
                level = "Low"

            records.append(
                {
                    "class": c,
                    "feature_index": m,
                    "SRCC": rho,
                    "abs_SRCC": abs_rho,
                    "sensitivity_level": level,
                    "p_value": pval,
                }
            )

    return pd.DataFrame(records)

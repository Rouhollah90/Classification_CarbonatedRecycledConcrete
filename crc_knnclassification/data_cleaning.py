"""Data cleaning utilities for CRC classification experiments."""

import numpy as np


def remove_duplicate_rows(x):
    x = np.asarray(x)
    return np.unique(x)


def trim_percentiles(x, lower=0.025, upper=0.975):
    x = np.asarray(x, dtype=float)
    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)
    keep_mask = (x >= q_low) & (x <= q_high)
    return x[keep_mask], keep_mask, q_low, q_high


def identify_extreme_percentiles(x, lower=0.05, upper=0.95):
    x = np.asarray(x, dtype=float)
    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)
    flag_mask = (x < q_low) | (x > q_high)
    return flag_mask, q_low, q_high


def winsorize_percentiles(x, lower=0.05, upper=0.95):
    x = np.asarray(x, dtype=float)
    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)
    return np.clip(x, q_low, q_high), q_low, q_high


def clean_distribution_data(
    x,
    trim_lower=0.025,
    trim_upper=0.975,
    winsor_lower=0.05,
    winsor_upper=0.95,
):
    x = remove_duplicate_rows(x)
    trimmed_x, _, q025, q975 = trim_percentiles(
        x,
        lower=trim_lower,
        upper=trim_upper,
    )
    flagged_mask, q05, q95 = identify_extreme_percentiles(
        trimmed_x,
        lower=winsor_lower,
        upper=winsor_upper,
    )
    winsorized_x, _, _ = winsorize_percentiles(
        trimmed_x,
        lower=winsor_lower,
        upper=winsor_upper,
    )

    report = {
        "Q025": q025,
        "Q975": q975,
        "Q05": q05,
        "Q95": q95,
        "n_original": len(x),
        "n_trimmed": len(trimmed_x),
        "n_flagged": int(np.sum(flagged_mask)),
        "trimmed_percent": 100 * (len(x) - len(trimmed_x)) / len(x),
    }
    return winsorized_x, report

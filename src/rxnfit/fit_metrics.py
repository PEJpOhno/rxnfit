# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

"""Compute RSS, TSS, and R² for model vs. experimental data.

TSS uses per-species means over the same valid points used for RSS.
All functions use the same valid-point definition (NaN excluded).
"""

import warnings
import numpy as np

from .expdata_reader import time_course, align_expdata_to_function_names


# Threshold below which TSS is considered nearly zero; callers may warn.
TSS_MIN_THRESHOLD = 1e-12


def expdata_df_to_datasets(expdata_df, function_names):
    """Convert a single experimental DataFrame to datasets format.

    Args:
        expdata_df (pandas.DataFrame): Time in first column, species
            concentrations in remaining columns. Column names must include
            all function_names.
        function_names (list[str]): Chemical species names (e.g. ODE order).

    Returns:
        list: One-element list of dicts with keys 't_list', 'C_exp_list'
            (aligned to function_names order; valid points only, NaN removed).

    Raises:
        ValueError: If a species in function_names is not in the DataFrame.
    """
    t_list, C_exp_list = time_course(expdata_df)
    columns = list(expdata_df.columns[1:])
    t_aligned, C_aligned = align_expdata_to_function_names(
        t_list, C_exp_list, columns, function_names
    )
    return [{"t_list": t_aligned, "C_exp_list": C_aligned}]


def _compute_tss(datasets):
    """Compute TSS using per-species means over all valid observations.

    Same valid points as used for RSS (each dataset's t_list[i], C_exp_list[i]).
    """
    if not datasets:
        return 0.0
    n_species = len(datasets[0]["C_exp_list"])
    all_C = [[] for _ in range(n_species)]
    for ds in datasets:
        for i in range(n_species):
            all_C[i].append(ds["C_exp_list"][i])
    # Flatten per species
    concat = []
    for i in range(n_species):
        concat.append(np.concatenate([np.ravel(c) for c in all_C[i]]))
    tss = 0.0
    for i in range(n_species):
        arr = concat[i]
        if arr.size == 0:
            continue
        mu = np.mean(arr)
        tss += np.sum((arr - mu) ** 2)
    return float(tss)


def fit_metrics(datasets, rss):
    """Compute fit metrics (RSS, TSS, R²) from datasets and RSS.

    TSS is computed from the same valid points as RSS, using per-species
    means. R² = 1 - RSS/TSS.

    Args:
        datasets (list[dict]): List of dicts with 't_list', 'C_exp_list'
            (each list of arrays in species order; valid points only).
        rss (float): Residual sum of squares (already computed).

    Returns:
        dict: Keys 'rss', 'tss', 'r2' (all float). If TSS is zero or very
            small, r2 may be inf or extreme; callers should warn when
            tss < TSS_MIN_THRESHOLD.
    """
    tss = _compute_tss(datasets)
    if tss <= 0:
        r2 = np.nan
    else:
        r2 = 1.0 - (float(rss) / tss)
    return {"rss": float(rss), "tss": float(tss), "r2": float(r2)}

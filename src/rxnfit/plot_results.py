# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

"""Matplotlib helpers for simulated or fitted concentration time courses.

Primary call sites are :meth:`rxnfit.solv_ode.RxnODEsolver.solution_plot` and
:meth:`rxnfit.expdata_fit.ExpDataFit.plot_fitted_solution`; the functions here
accept generic solution objects and optional experimental overlays.
"""

import warnings
from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .expdata_reader import get_time_unit_from_expdata


def plot_time_course_solutions(
    solution_list: List[Any],
    df_list: List[Optional[pd.DataFrame]],
    function_names: List[str],
    species: Optional[List[str]] = None,
    subplot_layout: Optional[Tuple[int, int]] = None,
    dataset_labels: Optional[List[str]] = None,
    show: bool = True,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Plot time-course solutions with optional experimental overlay.

    Draws one subplot per (solution, DataFrame) pair. Simulation curves and
    optional scatter points from DataFrames are shown. Optionally prints
    final concentrations.

    Args:
        solution_list: List of OdeResult or None. Length must match df_list
            when df_list is non-empty. When df_list is empty, uses
            solution_list as-is. None entries are skipped with a warning.
        df_list: List of DataFrames for experimental overlay. Can be empty
            (curves only). When non-empty, len must equal len(solution_list).
        function_names: List of species names in ODE order.
        species: Optional list of species to plot. If None, all species.
        subplot_layout: Optional (n_rows, n_cols). Default (n_plots, 1).
        dataset_labels: Optional list of labels for each subplot title.
            If None or not provided for an index, uses "Dataset {idx + 1}".
            Length should match the number of plotted datasets.
        show: If True, call plt.show() before returning. Default True.

    Returns:
        Tuple (fig, axes): The figure and axes array. Returns (None, None)
        if there are no successful integrations to plot.

    Raises:
        ValueError: If species names are invalid or list lengths mismatch.
    """
    all_species = list(function_names)
    if species is None:
        plot_species = all_species
    else:
        invalid = [s for s in species if s not in all_species]
        if invalid:
            raise ValueError(
                f"Species not in the ODE system: {invalid}. "
                f"Available: {all_species}"
            )
        plot_species = list(species)

    name_to_idx = {name: i for i, name in enumerate(all_species)}
    plot_items = [(name, name_to_idx[name]) for name in plot_species]

    # Pad df_list when empty
    if not df_list:
        effective_df_list = [None] * len(solution_list)
        time_unit = None
    else:
        if len(df_list) != len(solution_list):
            raise ValueError(
                f"Length mismatch: len(df_list)={len(df_list)}, "
                f"len(solution_list)={len(solution_list)}"
            )
        effective_df_list = df_list
        time_unit = get_time_unit_from_expdata(df_list)

    valid_pairs = [
        (idx, sol, plot_df)
        for idx, (sol, plot_df) in enumerate(zip(solution_list, effective_df_list))
        if sol is not None
    ]
    failed_pairs = [
        (idx, plot_df)
        for idx, (sol, plot_df) in enumerate(zip(solution_list, effective_df_list))
        if sol is None
    ]

    for idx, plot_df in failed_pairs:
        warnings.warn(
            f"Integration failed for dataset {idx + 1}.",
            UserWarning,
            stacklevel=4,
        )
        if plot_df is not None:
            print(f"  Failed DataFrame (dataset {idx + 1}):")
            print(plot_df.head())

    if not valid_pairs:
        print("No successful integrations to plot.")
        return (None, None)

    n_plots = len(valid_pairs)
    n_rows, n_cols = (
        subplot_layout if subplot_layout is not None else (n_plots, 1)
    )
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows)
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    t_min_all = min(float(sol.t.min()) for _, sol, _ in valid_pairs)
    t_max_all = max(float(sol.t.max()) for _, sol, _ in valid_pairs)

    print("\n=== Time-course plot ===")

    for plot_idx, (orig_idx, sol, plot_df) in enumerate(valid_pairs):
        ax = axes.flat[plot_idx]
        for species_name, i in plot_items:
            line, = ax.plot(
                sol.t, sol.y[i], label=species_name, linewidth=2
            )
            color = line.get_color()
            if plot_df is not None and species_name in plot_df.columns:
                t_exp = plot_df.iloc[:, 0].to_numpy()
                c_exp = plot_df[species_name].to_numpy()
                mask = ~np.isnan(c_exp)
                ax.scatter(
                    t_exp[mask], c_exp[mask],
                    color=color, marker='o', s=50, zorder=5,
                    edgecolors='white'
                )
        ax.set_xlim(t_min_all, t_max_all)
        xlabel = (
            f"Time ({time_unit})" if time_unit is not None
            else "Time ()"
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Concentration', fontsize=12)
        if dataset_labels is not None and orig_idx < len(dataset_labels):
            ax.set_title(dataset_labels[orig_idx], fontsize=14)
        else:
            ax.set_title(f'Dataset {orig_idx + 1}', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, axes.size):
        axes.flat[idx].set_visible(False)
    plt.tight_layout()
    if show:
        plt.show()

    print("\n=== Concentration at the final time point ===")
    for orig_idx, sol, _ in valid_pairs:
        if dataset_labels is not None and orig_idx < len(dataset_labels):
            label = dataset_labels[orig_idx]
        else:
            label = f"Dataset {orig_idx + 1}"
        print(f"{label}:")
        for name, i in plot_items:
            print(f"  {name}: {sol.y[i][-1]:.6f}")

    return (fig, axes)

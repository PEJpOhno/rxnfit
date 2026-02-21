# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 01/03/2026, M. Ohno

"""Load experimental time-course data from DataFrames.

Experimental data is passed as a list of DataFrames and returned as a list
of (t_list, C_exp_list) tuples. Functions handle missing values and align
data to ODE function names order.
"""

import pandas as pd


def time_course(df):
    """Extract time and concentration arrays from a DataFrame.

    The first column must contain time values. Missing values are removed
    per series (chemical species).

    Args:
        df (pandas.DataFrame): DataFrame with time in the first column and
            chemical species concentrations in subsequent columns.

    Returns:
        tuple:
            - t_list: List of time arrays, one per species. Each array
              contains only the time points where that species has valid data.
            - C_exp_list: List of concentration arrays, one per species.
              Each array contains only the valid concentration values.
    """
    time = df.iloc[:, 0].to_numpy()

    C_exp_list = []
    t_list = []

    for col in df.columns[1:]:
        c = df[col]
        mask = ~c.isna()          # Only non-missing values
        t_list.append(time[mask])
        C_exp_list.append(c[mask].to_numpy())

    return t_list, C_exp_list


def expdata_read(df_list):
    """Read experimental data from a list of DataFrames.

    Each DataFrame must have the same columns (time + species).
    Always returns a list, even for a single DataFrame.

    Args:
        df_list (list[pandas.DataFrame]): List of DataFrames containing
            experimental data. All DataFrames must have same column names.

    Returns:
        list[tuple]: List of (t_list, C_exp_list) tuples, one per DataFrame.
            Each t_list is a list of time arrays (one per species column).
            Each C_exp_list is a list of concentration arrays (one per
            species column). Missing values are removed per species.

    Raises:
        ValueError: If df_list is empty or if DataFrames have different
            column names.
    """
    if not df_list:
        raise ValueError("df_list cannot be empty.")

    base_cols = df_list[0].columns
    all_same = all(df.columns.equals(base_cols) for df in df_list)
    if not all_same:
        raise ValueError("All DataFrames must have the same column names.")

    datasets = []
    for df in df_list:
        t_list, C_exp_list = time_course(df)
        datasets.append((t_list, C_exp_list))

    return datasets


def get_y0_from_expdata(df_list, function_names):
    """Extract initial concentrations (t=0) from first row of each DataFrame.

    Returns concentrations in function_names order. For multiple DataFrames,
    returns a list of y0 vectors, one per DataFrame.

    Args:
        df_list (list[pandas.DataFrame]): List of DataFrames with time column
            and chemical species columns.
        function_names (list[str]): List of chemical species names in ODE
            variable order.

    Returns:
        list[list[float]]: List of initial concentration vectors, one per
            DataFrame. Each y0 is in function_names order. Missing values
            are filled with 0. len(return) == len(df_list).

    Raises:
        ValueError: If df_list is empty or a species name is not found
            in the DataFrame columns.
    """
    if not df_list:
        raise ValueError("df_list cannot be empty.")

    y0_list = []
    for df in df_list:
        first_row = df.iloc[0]
        y0 = []
        for name in function_names:
            if name in df.columns:
                val = first_row[name]
                y0.append(0.0 if pd.isna(val) else float(val))
            else:
                raise ValueError(
                    f"化学種 '{name}' がデータの列に見つかりません。"
                )
        y0_list.append(y0)

    return y0_list


def align_expdata_to_function_names(
        t_list, C_exp_list, columns, function_names):
    """Align time_course output to function_names order.

    time_course returns data in df.columns[1:] order. This function
    reorders to match ODE function_names.

    Args:
        t_list (list): List of time arrays from time_course
            (DataFrame column order).
        C_exp_list (list): List of concentration arrays from time_course.
        columns (list[str]): DataFrame column names (species only, no time).
        function_names (list[str]): Chemical species names in ODE order.

    Returns:
        tuple: (t_aligned, C_aligned) in function_names order.

    Raises:
        ValueError: If a species in function_names is not in columns.
    """
    col_to_idx = {name: i for i, name in enumerate(columns)}
    t_aligned = []
    C_aligned = []
    for name in function_names:
        i = col_to_idx.get(name)
        if i is None:
            raise ValueError(f"化学種 '{name}' がデータの列にありません。")
        t_aligned.append(t_list[i])
        C_aligned.append(C_exp_list[i])
    return t_aligned, C_aligned

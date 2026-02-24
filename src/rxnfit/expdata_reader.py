# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 01/03/2026, M. Ohno

"""Load experimental time-course data from DataFrames.

Experimental data is passed as a list of DataFrames and returned as a list
of (t_list, C_exp_list) tuples. Functions handle missing values and align
data to ODE function names order. Time column (0th column) unit and
consistency across DataFrames are also provided.
"""

import warnings
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


def get_time_unit_from_expdata(df_list):
    """Get time axis unit from the 0th column name and check consistency.

    The unit is derived from the first column name: if it contains '_',
    the part after the last '_' is used (e.g. "t_s" -> "s"); otherwise
    the full column name is used. When multiple DataFrames are given,
    their 0th column names must match; otherwise a warning is emitted
    and the first DataFrame's column is used.

    Args:
        df_list (list[pandas.DataFrame]): List of DataFrames with time
            in the first column. Can be a single DataFrame (wrapped in a list).

    Returns:
        str or None: The time unit string (e.g. "s", "min", "hr"), or None
            if df_list is empty.

    Raises:
        ValueError: If df_list is empty.
    """
    if not df_list:
        raise ValueError("df_list cannot be empty.")

    names_0 = [df.columns[0] for df in df_list]
    if len(set(names_0)) > 1:
        warnings.warn(
            f"経時変化データの0列目（時間列）の列名が一致しません: {names_0}. "
            "先頭のDataFrameの列名を用います。",
            UserWarning,
            stacklevel=2,
        )
    time_col_name = names_0[0]
    unit = (
        time_col_name.split("_")[-1]
        if "_" in str(time_col_name)
        else time_col_name
    )
    return unit


def get_t0_from_expdata(df_list):
    """Extract initial time from first row of each DataFrame.

    The first column is assumed to be time. Returns one float per DataFrame.

    Args:
        df_list (list[pandas.DataFrame]): List of DataFrames with time in
            the first column.

    Returns:
        list[float]: Initial time (first row, first column) for each
            DataFrame. len(return) == len(df_list).
    """
    if not df_list:
        raise ValueError("df_list cannot be empty.")
    return [float(df.iloc[0, 0]) for df in df_list]


def get_y0_from_expdata(df_list, function_names):
    """Extract initial concentrations from first row of each DataFrame.

    Returns concentrations in function_names order. For multiple DataFrames,
    returns a list of y0 vectors, one per DataFrame. The time of the first
    row can be obtained with get_t0_from_expdata; it is not required to be 0.

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

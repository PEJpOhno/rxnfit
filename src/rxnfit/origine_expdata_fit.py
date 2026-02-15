# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 01/03/2026, M. Ohno

"""
Load experimental time course concentration data and return them as a list
of DataFrames. Within each DataFrame, the measurement times are common to
all chemical species.
"""

import numpy as np
from scipy.integrate import solve_ivp
from sympy import Symbol
from sympy.core.symbol import Symbol as SympySymbol

from .build_ode import create_system_rhs


def time_course(df):
    """Extract time and concentration arrays from a DataFrame.

    The first column of the DataFrame must contain time values.
    Missing values are removed per series (chemical species).

    Args:
        df (pandas.DataFrame): DataFrame with time in the first column and
            chemical species concentrations in subsequent columns.

    Returns:
        tuple: A tuple containing:
            - t_list (list): List of time arrays, one per species. Each array
              contains only the time points where that species has valid data.
            - C_exp_list (list): List of concentration arrays, one per species.
              Each array contains only the valid concentration values for that
              species.
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
    Only the first DataFrame is processed, as all DataFrames should have
    the same structure.

    Args:
        df_list (list): List of DataFrames containing experimental data.
            All DataFrames must have the same column names.

    Returns:
        tuple: A tuple containing:
            - t_list (list): List of time arrays, one per species.
            - C_exp_list (list): List of concentration arrays, one per
              species.

    Raises:
        ValueError: If df_list is empty or if DataFrames have different
            column names.
    """
    # Check that all DataFrames have the same columns
    if not df_list:
        raise ValueError("df_list cannot be empty.")

    base_cols = df_list[0].columns
    all_same = all(df.columns.equals(base_cols) for df in df_list)
    if not all_same:
        raise ValueError("All DataFrames must have the same column names.")

    # Process the first DataFrame (all should have the same structure)
    t_list, C_exp_list = time_course(df_list[0])

    return t_list, C_exp_list


def solve_fit_model(builded_rxnode, fixed_initial_values):
    """Creates a model function for fitting only symbolic rate constants.

    The returned function solves the ODE system and returns predicted
    concentrations. Only symbolic rate constants are varied to match
    experimental data. Initial values and other numerically given
    experimental values are fixed; they are not optimized or changed.

    Args:
        builded_rxnode (RxnODEbuild): An instance of RxnODEbuild containing
            the reaction system definition.
        fixed_initial_values (list[float]): Initial concentrations for each
            species (in function_names order). Typically taken from
            experimental data at t=0. These values are never optimized.

    Returns:
        function: A function f(t, *params) where params are only the
            symbolic rate constant values (in symbolic_rate_const_keys order).
            The function signature is:
                f(t: array-like, *params: float) -> numpy.ndarray
            Returns a numpy.ndarray of shape (len(function_names), len(t))
            containing predicted concentrations for all species at each time
            point. The function has a `param_info` attribute containing:
            - symbolic_rate_consts: List of symbolic rate constant keys
            - function_names: List of chemical species names
            - n_params: Number of fitting parameters (symbolic rate constants)
            - fixed_initial_values: The fixed initial values used

    Raises:
        ValueError: If the length of fixed_initial_values does not match
            the number of chemical species (function_names).
    """
    ode_construct = builded_rxnode.get_ode_system()
    (system_of_equations, sympy_symbol_dict,
     ode_system, function_names, rate_consts_dict) = ode_construct

    ode_functions_with_rate_consts, symbolic_rate_const_keys = (
        builded_rxnode.create_ode_system_with_rate_consts()
    )

    fixed_rate_consts = {
        key: val for key, val in rate_consts_dict.items()
        if not isinstance(val, (Symbol, SympySymbol))
    }

    if len(fixed_initial_values) != len(function_names):
        raise ValueError(
            f"fixed_initial_valuesの長さ({len(fixed_initial_values)})が"
            f"化学種数({len(function_names)})と一致しません。"
        )

    y0_fixed = list(fixed_initial_values)

    def fit_function(t, *params):
        """Model function for ODE system with symbolic rate constants.

        Args:
            t (array-like): Time points at which to evaluate the solution.
            *params (float): Symbolic rate constant values in the order
                specified by symbolic_rate_const_keys.

        Returns:
            numpy.ndarray: Array of shape (len(function_names), len(t))
                containing predicted concentrations for all species at each
                time point.

        Raises:
            ValueError: If insufficient parameters are provided or if t is
                not an array-like object.
            RuntimeError: If numerical integration fails.
        """
        rate_const_values = dict(fixed_rate_consts)

        for i, key in enumerate(symbolic_rate_const_keys):
            if i < len(params):
                rate_const_values[key] = params[i]
            else:
                raise ValueError(
                    f"パラメータが不足しています。速度定数 {key} の値が必要です。"
                )

        system_rhs = create_system_rhs(
            ode_functions_with_rate_consts,
            function_names,
            rate_const_values=rate_const_values,
            symbolic_rate_const_keys=symbolic_rate_const_keys
        )

        if isinstance(t, (list, np.ndarray)):
            t_span = (t[0], t[-1])
            t_eval = np.array(t)
        else:
            raise ValueError("tは配列である必要があります")

        try:
            solution = solve_ivp(
                system_rhs,
                t_span,
                y0_fixed,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6
            )
            if not solution.success:
                raise RuntimeError(
                    f"数値積分が失敗しました: {solution.message}"
                )
            return solution.y
        except Exception as e:
            print(f"数値積分中にエラーが発生しました: {e}")
            raise

    fit_function.param_info = {
        'symbolic_rate_consts': symbolic_rate_const_keys,
        'function_names': function_names,
        'n_params': len(symbolic_rate_const_keys),
        'fixed_initial_values': fixed_initial_values,
    }

    return fit_function

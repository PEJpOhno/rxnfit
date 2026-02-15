# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 01/03/2026, M. Ohno

"""Load experimental data, extract initial values, and fit ODE parameters.

Experimental data is passed as a list of DataFrames and returned as a list
of (t_list, C_exp_list) tuples.
"""

import functools
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sympy import Symbol
from sympy.core.symbol import Symbol as SympySymbol

from .build_ode import create_system_rhs


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


def _align_expdata_to_function_names(
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


def _eval_ode_fit(t, *params, fit_ctx):
    """Evaluate ODE model for given time points and rate constant parameters.

    Core logic for single-dataset fitting. Used by solve_fit_model.

    Args:
        t (array-like): Time points at which to evaluate the solution.
        *params: Symbolic rate constant values in
            symbolic_rate_const_keys order.
        fit_ctx (dict): Context with fixed_rate_consts,
            symbolic_rate_const_keys, ode_functions_with_rate_consts,
            function_names, y0_fixed, t_span, method, rtol.

    Returns:
        numpy.ndarray: Predicted concentrations,
            shape (len(function_names), len(t)).

    Raises:
        ValueError: If insufficient parameters or t is not array-like.
        RuntimeError: If numerical integration fails.
    """
    fixed_rate_consts = fit_ctx['fixed_rate_consts']
    symbolic_rate_const_keys = fit_ctx['symbolic_rate_const_keys']
    ode_functions_with_rate_consts = fit_ctx['ode_functions_with_rate_consts']
    function_names = fit_ctx['function_names']
    y0_fixed = fit_ctx['y0_fixed']
    t_span = fit_ctx['t_span']
    method = fit_ctx['method']
    rtol = fit_ctx['rtol']

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
        t_eval = np.array(t)
    else:
        raise ValueError("tは配列である必要があります")

    try:
        solution = solve_ivp(
            system_rhs,
            t_span,
            y0_fixed,
            t_eval=t_eval,
            method=method,
            rtol=rtol
        )
        if not solution.success:
            raise RuntimeError(
                f"数値積分が失敗しました: {solution.message}"
            )
        return solution.y
    except Exception as e:
        print(f"数値積分中にエラーが発生しました: {e}")
        raise


def _compute_multi_residual(params, fit_ctx):
    """Compute sum of squared residuals across all datasets.

    Core logic for multi-dataset fitting. Used by solve_fit_model_multi.

    Args:
        params (list): Symbolic rate constant values in
            symbolic_rate_const_keys order.
        fit_ctx (dict): Context with fixed_rate_consts,
            symbolic_rate_const_keys, ode_functions_with_rate_consts,
            function_names, datasets,
            t_span, method, rtol.

    Returns:
        float: Sum of squared residuals, or np.inf on integration failure.
    """
    fixed_rate_consts = fit_ctx['fixed_rate_consts']
    symbolic_rate_const_keys = fit_ctx['symbolic_rate_const_keys']
    ode_functions_with_rate_consts = fit_ctx['ode_functions_with_rate_consts']
    function_names = fit_ctx['function_names']
    datasets = fit_ctx['datasets']
    t_span = fit_ctx['t_span']
    method = fit_ctx['method']
    rtol = fit_ctx['rtol']

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

    total_residual = 0.0
    for ds in datasets:
        y0 = ds['y0']
        t_list = ds['t_list']
        C_exp_list = ds['C_exp_list']

        t_all = np.unique(np.concatenate(t_list))

        try:
            solution = solve_ivp(
                system_rhs,
                t_span,
                y0,
                t_eval=t_all,
                method=method,
                rtol=rtol
            )
            if not solution.success:
                return np.inf
        except Exception:
            return np.inf

        for i in range(len(function_names)):
            t_i = t_list[i]
            C_i = C_exp_list[i]
            if len(t_i) == 0:
                continue
            y_model_i = np.interp(t_i, solution.t, solution.y[i])
            total_residual += np.sum((y_model_i - C_i) ** 2)

    return total_residual


def solve_fit_model(
        builded_rxnode, fixed_initial_values, t_span,
        method="RK45", rtol=1e-6):
    """Create a model function for fitting only symbolic rate constants.

    The returned function solves the ODE system and returns predicted
    concentrations. Only symbolic rate constants are varied; initial values
    and other numeric values are fixed.

    Args:
        builded_rxnode (RxnODEbuild): Instance containing the reaction
            system definition.
        fixed_initial_values (list[float]): Initial concentrations for each
            species (in function_names order). Typically from experimental
            data at t=0. These values are never optimized.
        t_span (tuple[float, float]): Integration time span (t_start, t_end).
            Must encompass the experimental time points. Required.
        method (str, optional): Integration method for solve_ivp.
            Defaults to "RK45".
        rtol (float, optional): Relative tolerance for solve_ivp.
            Defaults to 1e-6.

    Returns:
        callable: A function f(t, *params) where params are symbolic rate
            constant values in symbolic_rate_const_keys order.
            Returns
            numpy.ndarray of shape (len(function_names), len(t)). Has
            param_info attribute with: symbolic_rate_consts, function_names,
            n_params, fixed_initial_values.

    Raises:
        ValueError: If len(fixed_initial_values) != number of chemical species.
    """
    ode_construct = builded_rxnode.get_ode_system()
    (_, _, _, function_names, rate_consts_dict) = ode_construct

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

    fit_ctx = {
        'fixed_rate_consts': fixed_rate_consts,
        'symbolic_rate_const_keys': symbolic_rate_const_keys,
        'ode_functions_with_rate_consts': ode_functions_with_rate_consts,
        'function_names': function_names,
        'y0_fixed': y0_fixed,
        't_span': t_span,
        'method': method,
        'rtol': rtol,
    }
    fit_function = functools.partial(_eval_ode_fit, fit_ctx=fit_ctx)
    fit_function.param_info = {
        'symbolic_rate_consts': symbolic_rate_const_keys,
        'function_names': function_names,
        'n_params': len(symbolic_rate_const_keys),
        'fixed_initial_values': fixed_initial_values,
    }

    return fit_function


def solve_fit_model_multi(
        builded_rxnode, df_list, t_span,
        method="RK45", rtol=1e-6):
    """Create residual function for multi-dataset fitting (varying y0).

    y0 for each dataset is taken from the first row (t=0) of each DataFrame.
    Only rate constants are optimized; minimizes sum of residuals across
    all datasets.

    Args:
        builded_rxnode (RxnODEbuild): Instance containing the reaction
            system definition.
        df_list (list[pandas.DataFrame]): List of experimental DataFrames.
            All must have same structure (time + species columns).
        t_span (tuple[float, float]): Integration time span (t_start, t_end).
            Must encompass all experimental time points. Required.
        method (str, optional): Integration method for solve_ivp.
            Defaults to "RK45".
        rtol (float, optional): Relative tolerance for solve_ivp.
            Defaults to 1e-6.

    Returns:
        tuple:
            - residual_func: Callable that takes params
              (list of symbolic rate constant values in
              symbolic_rate_const_keys order) and returns
              scalar residual (sum of squared residuals).
            - param_info: Dict with symbolic_rate_consts, function_names,
              n_params, n_datasets, y0_list.

    Raises:
        ValueError: If df_list is empty or column structure is invalid.
    """
    if not df_list:
        raise ValueError("df_list cannot be empty.")

    ode_construct = builded_rxnode.get_ode_system()
    (_, _, _, function_names, rate_consts_dict) = ode_construct

    ode_functions_with_rate_consts, symbolic_rate_const_keys = (
        builded_rxnode.create_ode_system_with_rate_consts()
    )

    fixed_rate_consts = {
        key: val for key, val in rate_consts_dict.items()
        if not isinstance(val, (Symbol, SympySymbol))
    }

    # データセットの読み込みと整列
    datasets_raw = expdata_read(df_list)
    y0_list = get_y0_from_expdata(df_list, function_names)
    columns = list(df_list[0].columns[1:])  # 化学種列のみ

    datasets = []
    for (t_list, C_exp_list), y0 in zip(datasets_raw, y0_list):
        t_aligned, C_aligned = _align_expdata_to_function_names(
            t_list, C_exp_list, columns, function_names
        )
        datasets.append({
            'y0': y0, 't_list': t_aligned, 'C_exp_list': C_aligned
        })

    fit_ctx = {
        'fixed_rate_consts': fixed_rate_consts,
        'symbolic_rate_const_keys': symbolic_rate_const_keys,
        'ode_functions_with_rate_consts': ode_functions_with_rate_consts,
        'function_names': function_names,
        'datasets': datasets,
        't_span': t_span,
        'method': method,
        'rtol': rtol,
    }
    residual_func = functools.partial(_compute_multi_residual, fit_ctx=fit_ctx)

    param_info = {
        'symbolic_rate_consts': symbolic_rate_const_keys,
        'function_names': function_names,
        'n_params': len(symbolic_rate_const_keys),
        'n_datasets': len(datasets),
        'y0_list': y0_list,
    }

    return residual_func, param_info


class ExpDataFitSci:
    """Multi-dataset fitting of symbolic rate constants to experimental data.

    Fits ODE rate constants using experimental time course data.
    Provides methods to run fitting and to prepare arguments for
    solv_ode (RxnODEsolver) for re-analysis with fitted parameters.
    """

    def __init__(self, builded_rxnode, df_list, t_range,
                 method="RK45", rtol=1e-6):
        """Initialize the fitting context.

        Args:
            builded_rxnode: RxnODEbuild instance (reaction system).
            df_list (list): List of experimental DataFrames.
            t_range (tuple[float, float]): Integration time span
                (t_start, t_end). Required.
            method (str, optional): solve_ivp method. Defaults to "RK45".
            rtol (float, optional): solve_ivp relative tolerance.
                Defaults to 1e-6.
        """
        self.builded_rxnode = builded_rxnode
        self.df_list = df_list
        self.t_range = t_range
        self.method = method
        self.rtol = rtol
        self._param_info = None
        self._result = None

    def run_fit(self, p0, opt_method='L-BFGS-B', bounds=None, verbose=True):
        """Run fitting and return optimized rate constants.

        Args:
            p0 (list): Initial guess for symbolic rate constants
                (in symbolic_rate_const_keys order).
            opt_method (str, optional): scipy.optimize.minimize method.
                Defaults to 'L-BFGS-B'.
            bounds (list, optional): Bounds for each parameter. If None,
                uses [(1e-10, None)] * n_params.
            verbose (bool, optional): Print optimization result.
                Defaults True.

        Returns:
            tuple: (result, param_info)
                - result: scipy.optimize.OptimizeResult.
                - param_info: Dict with symbolic_rate_consts, etc.
        """
        residual_func, param_info = solve_fit_model_multi(
            self.builded_rxnode, self.df_list, self.t_range,
            method=self.method, rtol=self.rtol
        )

        n_params = param_info['n_params']
        if len(p0) != n_params:
            raise ValueError(
                f"p0の長さ({len(p0)})がシンボリック速度定数の数"
                f"({n_params})と一致しません。"
            )

        if bounds is None:
            bounds = [(1e-10, None)] * n_params

        result = minimize(
            residual_func,
            p0,
            method=opt_method,
            bounds=bounds,
        )

        self._param_info = param_info
        self._result = result

        if verbose:
            symbolic_keys = param_info['symbolic_rate_consts']
            print(f"最適化成功: {result.success}")
            print("最適化された速度定数:")
            for k, v in zip(symbolic_keys, result.x):
                print(f"  {k} = {v:.6g}")
            print(f"残差二乗和: {result.fun:.6g}")

        return result, param_info

    def get_solver_config_args(self, dataset_index=0):
        """Get kwargs for SolverConfig for use with RxnODEsolver.

        Use after run_fit. Provides y0, t_span, method, rtol for
        the specified dataset.

        Args:
            dataset_index (int, optional): Index of dataset for y0.
                Defaults to 0 (first dataset).

        Returns:
            dict: Keyword args for SolverConfig:
                y0, t_span, method, rtol.

        Raises:
            RuntimeError: If run_fit has not been called.
        """
        if self._param_info is None:
            raise RuntimeError("run_fit を先に実行してください。")
        y0 = self._param_info['y0_list'][dataset_index]
        return {
            'y0': y0,
            't_span': self.t_range,
            'method': self.method,
            'rtol': self.rtol,
        }

    def get_fitted_rate_const_dict(self, result=None):
        """Get dict of fitted rate constants for builder.rate_consts_dict.

        Use to update builded_rxnode.rate_consts_dict before passing
        to RxnODEsolver.

        Args:
            result: OptimizeResult from run_fit. If None, uses internal
                result from last run_fit.

        Returns:
            dict: Fitted rate constants {key: value} to merge into
                builded_rxnode.rate_consts_dict.

        Raises:
            RuntimeError: If result is None and run_fit not yet called.
        """
        res = result if result is not None else self._result
        if res is None or self._param_info is None:
            raise RuntimeError("run_fit を先に実行してください。")
        symbolic_keys = self._param_info['symbolic_rate_consts']
        return dict(zip(symbolic_keys, res.x))


def run_fit_multi(builded_rxnode, df_list, p0, t_range=None,
                  method="RK45", rtol=1e-6,
                  opt_method='L-BFGS-B', bounds=None, verbose=True):
    """Convenience wrapper around ExpDataFitSci.run_fit.

    If t_range is None, derives from first DataFrame.
    """
    if t_range is None:
        t_range = (
            float(df_list[0].iloc[:, 0].min()),
            float(df_list[0].iloc[:, 0].max())
        )
    fit_sci = ExpDataFitSci(
        builded_rxnode, df_list, t_range, method=method, rtol=rtol
    )
    return fit_sci.run_fit(
        p0, opt_method=opt_method, bounds=bounds, verbose=verbose
    )

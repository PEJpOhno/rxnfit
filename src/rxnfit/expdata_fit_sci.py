# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 01/03/2026, M. Ohno

"""Fit ODE parameters to experimental time-course data.

This module provides functions and classes for fitting symbolic rate constants
in ODE systems to experimental data using scipy.optimize.minimize.
"""

import functools
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, OptimizeResult
from sympy import Symbol
from sympy.core.symbol import Symbol as SympySymbol

from .build_ode import create_system_rhs
from .expdata_reader import (
    expdata_read,
    get_t0_from_expdata,
    get_y0_from_expdata,
    align_expdata_to_function_names
)
from .fit_metrics import fit_metrics as compute_fit_metrics
from .fit_metrics import TSS_MIN_THRESHOLD
from .rate_const_ft_eval import has_time_dependent_rates, build_evaluator
from .plot_results import plot_time_course_solutions
from .solv_ode import _ode_result_to_dataframe

from typing import Optional, List, Union

import pandas as pd


def _resolve_df_names(df_list, df_names=None):
    """Resolve dataset names from df_names or DataFrame.attrs['name'].

    When df_names is None, uses df.attrs.get('name') for each DataFrame if
    present and non-empty; otherwise falls back to str(i).

    Args:
        df_list: List of DataFrames.
        df_names: Optional list of names. If provided, must match len(df_list).

    Returns:
        list[str]: Resolved names for each DataFrame.

    Raises:
        ValueError: If df_names is provided and len(df_names) != len(df_list).
    """
    if df_names is not None:
        if len(df_names) != len(df_list):
            raise ValueError(
                f"len(df_names)={len(df_names)} must match len(df_list)={len(df_list)}."
            )
        return list(df_names)
    result = []
    for i, df in enumerate(df_list):
        name = getattr(df, 'attrs', {}).get('name')
        if isinstance(name, str) and name.strip():
            result.append(name.strip())
        else:
            result.append(str(i))
    return result


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
                f"Insufficient parameters. A value for rate constant {key} is required."
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
        raise ValueError("t must be an array.")

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
                f"Numerical integration failed: {solution.message}"
            )
        return solution.y
    except Exception as e:
        print(f"An error occurred during numerical integration: {e}")
        raise


def _integrate_datasets_for_params(fit_ctx, params):
    """Integrate ODE for each dataset with given params.

    Args:
        fit_ctx (dict): Context with fixed_rate_consts, symbolic_rate_const_keys,
            ode_functions_with_rate_consts, function_names, datasets,
            t_span, method, rtol. Optionally 'evaluator' for k(t).
        params (list): Symbolic rate constant values in symbolic_rate_const_keys order.

    Returns:
        list: Length len(fit_ctx['datasets']). Each element is OdeResult or None.
            None for failed integrations.
    """
    fixed_rate_consts = fit_ctx['fixed_rate_consts']
    symbolic_rate_const_keys = fit_ctx['symbolic_rate_const_keys']
    ode_functions_with_rate_consts = fit_ctx['ode_functions_with_rate_consts']
    function_names = fit_ctx['function_names']
    datasets = fit_ctx['datasets']
    t_span = fit_ctx['t_span']
    method = fit_ctx['method']
    rtol = fit_ctx['rtol']

    if 'evaluator' in fit_ctx:
        evaluator = fit_ctx['evaluator']
        rate_const_values = lambda t: evaluator(t, params)
    else:
        rate_const_values = dict(fixed_rate_consts)
        for i, key in enumerate(symbolic_rate_const_keys):
            if i < len(params):
                rate_const_values[key] = params[i]
            else:
                raise ValueError(
                    f"Insufficient parameters. A value for rate constant {key} is required."
                )

    system_rhs = create_system_rhs(
        ode_functions_with_rate_consts,
        function_names,
        rate_const_values=rate_const_values,
        symbolic_rate_const_keys=symbolic_rate_const_keys
    )

    solution_list = []
    for ds in datasets:
        y0 = ds['y0']
        t_list = ds['t_list']
        t_exp = np.unique(np.concatenate(t_list))
        t_start = ds['t0']
        t_span_ds = (t_start, t_span[1])
        # プロットの角張りを防ぐため、密な t_eval を使用（残差計算は np.interp で補間）
        n_dense = max(100, len(t_exp) * 10)
        t_dense = np.linspace(t_start, t_span[1], n_dense)
        t_eval = np.sort(np.unique(np.concatenate([t_exp, t_dense])))

        try:
            solution = solve_ivp(
                system_rhs,
                t_span_ds,
                y0,
                t_eval=t_eval,
                method=method,
                rtol=rtol
            )
            if not solution.success:
                solution_list.append(None)
            else:
                solution_list.append(solution)
        except Exception:
            solution_list.append(None)

    return solution_list


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
    function_names = fit_ctx['function_names']
    datasets = fit_ctx['datasets']

    solution_list = _integrate_datasets_for_params(fit_ctx, params)
    if any(sol is None for sol in solution_list):
        return np.inf

    total_residual = 0.0
    for solution, ds in zip(solution_list, datasets):
        t_list = ds['t_list']
        C_exp_list = ds['C_exp_list']
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
            f"len(fixed_initial_values) ({len(fixed_initial_values)}) does not "
            f"match the number of species ({len(function_names)})."
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
        method="RK45", rtol=1e-6, df_names=None):
    """Create residual function for multi-dataset fitting (varying y0).

    y0 and the initial time (t0) for each dataset are taken from the first
    row of each DataFrame. Integration runs from that t0 to t_span[1].
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
        df_names (list[str], optional): Names for each DataFrame. If None,
            uses df.attrs.get('name') for each when present and non-empty;
            otherwise str(i). Length must match len(df_list) when provided.

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

    resolved_df_names = _resolve_df_names(df_list, df_names)

    ode_construct = builded_rxnode.get_ode_system()
    (_, _, _, function_names, rate_consts_dict) = ode_construct

    ode_functions_with_rate_consts, symbolic_rate_const_keys = (
        builded_rxnode.create_ode_system_with_rate_consts()
    )

    fixed_rate_consts = {
        key: val for key, val in rate_consts_dict.items()
        if isinstance(val, (int, float))
    }
    # データセットの読み込みと整列
    datasets_raw = expdata_read(df_list)
    y0_list = get_y0_from_expdata(df_list, function_names)
    t0_list = get_t0_from_expdata(df_list)
    columns = list(df_list[0].columns[1:])

    datasets = []
    for (t_list, C_exp_list), y0, t0 in zip(datasets_raw, y0_list, t0_list):
        t_aligned, C_aligned = align_expdata_to_function_names(
            t_list, C_exp_list, columns, function_names
        )
        datasets.append({
            'y0': y0, 't0': t0, 't_list': t_aligned, 'C_exp_list': C_aligned
        })

    fit_ctx = {
        'fixed_rate_consts': fixed_rate_consts,
        'symbolic_rate_const_keys': symbolic_rate_const_keys,
        'ode_functions_with_rate_consts': ode_functions_with_rate_consts,
        'function_names': function_names,
        'datasets': datasets,
        'df_names': resolved_df_names,
        't_span': t_span,
        'method': method,
        'rtol': rtol,
    }
    if has_time_dependent_rates(rate_consts_dict):
        fit_ctx['evaluator'] = build_evaluator(
            rate_consts_dict,
            symbolic_rate_const_keys,
            fixed_rate_consts,
        )

    residual_func = functools.partial(_compute_multi_residual, fit_ctx=fit_ctx)

    param_info = {
        'symbolic_rate_consts': symbolic_rate_const_keys,
        'function_names': function_names,
        'n_params': len(symbolic_rate_const_keys),
        'n_datasets': len(datasets),
        'y0_list': y0_list,
        't0_list': t0_list,
    }

    return residual_func, param_info, fit_ctx


def _normalize_p0(p0, param_info):
    """Convert p0 to a list in symbolic_rate_const_keys order.

    If p0 is a dict, validate keys (no extra, no missing) and values (numeric),
    then return a list in param_info['symbolic_rate_consts'] order.
    If p0 is a list or tuple, check length and return as list.

    Raises:
        ValueError: If p0 is dict with invalid keys or non-numeric values,
            or if sequence length does not match n_params.
    """
    symbolic_keys = param_info['symbolic_rate_consts']
    n_params = param_info['n_params']

    if isinstance(p0, dict):
        p0_keys = set(p0.keys())
        allowed = set(symbolic_keys)
        extra = p0_keys - allowed
        if extra:
            raise ValueError(
                f"p0 contains undefined symbolic rate constant names: {sorted(extra)}. "
                f"Valid keys: {sorted(allowed)}."
            )
        missing = allowed - p0_keys
        if missing:
            raise ValueError(
                f"p0 is missing the following symbolic rate constants: {sorted(missing)}."
            )
        try:
            return [float(p0[k]) for k in symbolic_keys]
        except (TypeError, ValueError):
            raise ValueError("All values in p0 must be numeric.")
    else:
        p0_list = list(p0)
        if len(p0_list) != n_params:
            raise ValueError(
                f"p0 length ({len(p0_list)}) does not match number of symbolic "
                f"rate constants ({n_params})."
            )
        return p0_list


class ExpDataFitSci:
    """Multi-dataset fitting of symbolic rate constants to experimental data.

    Fits ODE rate constants using experimental time course data.
    Provides methods to run fitting and to prepare arguments for
    solv_ode (RxnODEsolver) for re-analysis with fitted parameters.
    """

    def __init__(self, builded_rxnode, df_list, t_range,
                 method="RK45", rtol=1e-6, df_names=None):
        """Initialize the fitting context.

        Args:
            builded_rxnode (RxnODEbuild): Instance containing the reaction
                system definition.
            df_list (list[pandas.DataFrame]): List of experimental DataFrames.
                All must have same structure (time + species columns).
            t_range (tuple[float, float]): Integration time span
                (t_start, t_end). Required.
            method (str, optional): Integration method for solve_ivp.
                Defaults to "RK45".
            rtol (float, optional): Relative tolerance for solve_ivp.
                Defaults to 1e-6.
            df_names (list[str], optional): Names for each DataFrame, used by
                plot_fitted_solution for plot_datasets selection and subplot
                titles. If None, uses df.attrs.get('name') for each DataFrame
                when present and non-empty; otherwise falls back to str(i).
                Length must match len(df_list) when provided.
        """
        self.builded_rxnode = builded_rxnode
        self.df_list = df_list
        self.t_range = t_range
        self.method = method
        self.rtol = rtol
        self.df_names = _resolve_df_names(df_list, df_names)
        self._param_info = None
        self._result = None
        self._fit_ctx = None

    def run_fit(self, p0, opt_method='L-BFGS-B', bounds=None, verbose=True,
                use_log_fit=False, lower_bound=None):
        """Run fitting and return optimized rate constants.

        Args:
            p0 (list, tuple, or dict): Initial guess for symbolic rate constants.
                - list or tuple: Values in the order of symbolic_rate_const_keys.
                  The order you give is assumed correct. Length must match the
                  number of symbolic rate constants.
                - dict: Keys are symbolic rate constant names (strings, e.g.
                  "k1", "k2"). Values are initial guesses (numeric). Keys must
                  match get_symbolic_rate_const_keys() exactly (no extra keys,
                  no missing keys). Example: {"k1": 0.001, "k2": 0.002}.
                When use_log_fit=True, all values must be positive.
            opt_method (str, optional): scipy.optimize.minimize method.
                Defaults to 'L-BFGS-B'.
            bounds (list, optional): Bounds for each parameter (linear fit only),
                in symbolic_rate_const_keys order. If None, uses
                [(lower_bound or 1e-10, None)] * n_params. When given,
                lower_bound is ignored. When use_log_fit=True, bounds is
                ignored; lower_bound (or default 1e-6) is used instead.
                Defaults to None.
            verbose (bool, optional): Print optimization result.
                Defaults to True.
            use_log_fit (bool, optional): If True, optimize in log(k) space for
                numerical stability with small rate constants. result.x is
                always returned in linear scale (k). Defaults to False.
            lower_bound (float, optional): Common lower bound for all parameters
                (must be positive). When None: linear fit uses 1e-10, log fit
                uses 1e-6. Defaults to None.

        Returns:
            tuple: (result, param_info, fit_metrics)
                - result: Object with .x (linear-scale k), .success, .fun,
                  .tss, .r2. (RSS is .fun; do not use .rss.)
                - param_info: Dict with symbolic_rate_consts, etc.
                - fit_metrics: Dict with keys 'rss', 'tss', 'r2'.

        Raises:
            ValueError: If p0 has wrong length (sequence), invalid keys or
                non-numeric values (dict), lower_bound <= 0, or
                use_log_fit=True with non-positive p0 values.
        """
        residual_func, param_info, fit_ctx = solve_fit_model_multi(
            self.builded_rxnode, self.df_list, self.t_range,
            method=self.method, rtol=self.rtol, df_names=self.df_names
        )
        self._fit_ctx = fit_ctx

        n_params = param_info['n_params']
        p0 = _normalize_p0(p0, param_info)

        if lower_bound is not None and lower_bound <= 0:
            raise ValueError(
                "lower_bound must be positive."
            )

        if use_log_fit:
            p0_arr = np.asarray(p0, dtype=float)
            if np.any(p0_arr <= 0):
                raise ValueError(
                    "When use_log_fit=True, all elements of p0 must be positive."
                )
            low = lower_bound if lower_bound is not None else 1e-6
            p0_log = np.log(p0_arr)
            bounds_log = [(np.log(low), None)] * n_params

            def residual_log_safe(p):
                r = residual_func(np.exp(p))
                return r if np.isfinite(r) else 1e15

            result_log = minimize(
                residual_log_safe,
                p0_log,
                method=opt_method,
                bounds=bounds_log,
            )
            x_linear = np.exp(result_log.x)
            result = OptimizeResult(
                x=x_linear,
                success=result_log.success,
                fun=result_log.fun,
                message=getattr(result_log, 'message', ''),
                nfev=getattr(result_log, 'nfev', None),
                nit=getattr(result_log, 'nit', None),
                njev=getattr(result_log, 'njev', None),
            )
            self._param_info = param_info
            self._result = result

            metrics = compute_fit_metrics(fit_ctx['datasets'], result.fun)
            result.tss = metrics['tss']
            result.r2 = metrics['r2']
            if metrics['tss'] < TSS_MIN_THRESHOLD:
                warnings.warn(
                    "TSS is nearly zero; R² may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
            if verbose:
                symbolic_keys = param_info['symbolic_rate_consts']
                print(f"Optimization success: {result.success}")
                print("Fitted rate constants:")
                for k, v in zip(symbolic_keys, x_linear):
                    print(f"  {k} = {v:.6g}")
                print(
                    f"Residual sum of squares: {metrics['rss']:.6g}  "
                    f"R²: {metrics['r2']:.6g}"
                )

            return result, param_info, metrics

        # Linear fit
        if bounds is None:
            bounds = [(lower_bound or 1e-10, None)] * n_params

        result = minimize(
            residual_func,
            p0,
            method=opt_method,
            bounds=bounds,
        )

        self._param_info = param_info
        self._result = result

        metrics = compute_fit_metrics(fit_ctx['datasets'], result.fun)
        result.tss = metrics['tss']
        result.r2 = metrics['r2']
        if metrics['tss'] < TSS_MIN_THRESHOLD:
            warnings.warn(
                "TSS is nearly zero; R² may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        if verbose:
            symbolic_keys = param_info['symbolic_rate_consts']
            print(f"Optimization success: {result.success}")
            print("Fitted rate constants:")
            for k, v in zip(symbolic_keys, result.x):
                print(f"  {k} = {v:.6g}")
            print(
                f"Residual sum of squares: {metrics['rss']:.6g}  "
                f"R²: {metrics['r2']:.6g}"
            )

        return result, param_info, metrics

    def get_solver_config_args(self, dataset_index=0):
        """Get kwargs for SolverConfig for use with RxnODEsolver.

        Use after run_fit. Provides y0, t_span, method, rtol for
        the specified dataset. When the model has time-dependent rate
        constants k(t), also adds rate_const_values (callable) and
        symbolic_rate_const_keys; in that case run_fit() must have been
        called first.

        Args:
            dataset_index (int, optional): Index of dataset for y0.
                Defaults to 0 (first dataset).

        Returns:
            dict: Keyword args for SolverConfig:
                y0, t_span, method, rtol, and optionally
                rate_const_values, symbolic_rate_const_keys when k(t) present.

        Raises:
            RuntimeError: If run_fit not executed.
        """
        if self._param_info is None:
            raise RuntimeError("run_fit not executed")
        y0 = self._param_info['y0_list'][dataset_index]
        t0 = self._param_info['t0_list'][dataset_index]
        out = {
            'y0': y0,
            't_span': (t0, self.t_range[1]),
            'method': self.method,
            'rtol': self.rtol,
        }
        if has_time_dependent_rates(self.builded_rxnode.rate_consts_dict):
            if self._result is None:
                raise RuntimeError("run_fit not executed")
            fixed_rate_consts = {
                k: float(v) for k, v in self.builded_rxnode.rate_consts_dict.items()
                if isinstance(v, (int, float))
            }
            evaluator = build_evaluator(
                self.builded_rxnode.rate_consts_dict,
                self._param_info['symbolic_rate_consts'],
                fixed_rate_consts,
            )
            out['rate_const_values'] = lambda t: evaluator(t, self._result.x)
            out['symbolic_rate_const_keys'] = self._param_info['symbolic_rate_consts']
        return out

    def get_fitted_rate_const_dict(self, result=None):
        """Get dict of fitted rate constants for builder.rate_consts_dict.

        Use to update builded_rxnode.rate_consts_dict before passing
        to RxnODEsolver.

        Args:
            result (scipy.optimize.OptimizeResult, optional): OptimizeResult
                from run_fit. If None, uses internal result from last run_fit.
                Defaults to None.

        Returns:
            dict: Fitted rate constants {key: value} to merge into
                builded_rxnode.rate_consts_dict.

        Raises:
            RuntimeError: If run_fit not executed.
        """
        res = result if result is not None else self._result
        if res is None or self._param_info is None:
            raise RuntimeError("run_fit not executed")
        symbolic_keys = self._param_info['symbolic_rate_consts']
        return dict(zip(symbolic_keys, res.x))

    def plot_fitted_solution(
        self,
        expdata_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        plot_datasets: Optional[List[str]] = None,
        species: Optional[List[str]] = None,
        subplot_layout: Optional[tuple] = None,
    ):
        """Plot fitted time-course with per-dataset y0.

        Use after run_fit. Each subplot uses the y0 from the corresponding
        DataFrame's first row. When expdata_df is None, uses self.df_list
        for experimental overlay. When plot_datasets is given, only those
        datasets are plotted (by df_names). Subplot titles use dataset names
        (df_names) when available; otherwise "Dataset 1", "Dataset 2", ...

        Args:
            expdata_df: Experimental data for overlay. Single DataFrame or
                list. If None, uses self.df_list (fit data). When
                plot_datasets is given, length must match len(plot_datasets);
                otherwise must match n_datasets.
            plot_datasets: List of dataset names (from df_names) to plot.
                If None, all datasets are plotted.
            species: Species to plot. If None, all.
            subplot_layout: (n_rows, n_cols) for subplot grid.

        Returns:
            None

        Raises:
            RuntimeError: If run_fit not executed.
            ValueError: If length/column mismatch or unknown plot_datasets name.
        """
        if self._param_info is None or self._result is None or self._fit_ctx is None:
            raise RuntimeError("run_fit not executed")

        n_datasets = len(self._fit_ctx['datasets'])
        df_names = self._fit_ctx.get(
            'df_names', [str(i) for i in range(n_datasets)]
        )

        # プロット対象のインデックス
        if plot_datasets is not None:
            name_to_idx = {name: i for i, name in enumerate(df_names)}
            for name in plot_datasets:
                if name not in name_to_idx:
                    raise ValueError(
                        f"plot_datasets: '{name}' not in df_names {df_names}"
                    )
            plot_indices = [name_to_idx[name] for name in plot_datasets]
        else:
            plot_indices = list(range(n_datasets))

        n_plot = len(plot_indices)

        # 実験データのオーバーレイ用 df_list
        if expdata_df is None:
            df_list_filtered = [self.df_list[i] for i in plot_indices]
        else:
            df_list_arg = expdata_df if isinstance(expdata_df, list) else [expdata_df]
            if len(df_list_arg) != n_plot:
                raise ValueError(
                    f"Length mismatch: expdata_df has {len(df_list_arg)} "
                    f"elements but {n_plot} datasets to plot."
                )
            df_list_filtered = df_list_arg
        for df in df_list_filtered:
            for name in self._param_info['function_names']:
                if name not in df.columns:
                    raise ValueError(
                        f"Column '{name}' not found in DataFrame. "
                        f"Required: {self._param_info['function_names']}"
                    )

        solution_list_full = _integrate_datasets_for_params(
            self._fit_ctx, list(self._result.x)
        )
        solution_list = [solution_list_full[i] for i in plot_indices]
        dataset_labels = [df_names[i] for i in plot_indices]

        plot_time_course_solutions(
            solution_list,
            df_list_filtered,
            self._param_info['function_names'],
            species=species,
            subplot_layout=subplot_layout,
            dataset_labels=dataset_labels,
        )

    def to_dataframe_list(self, time_column_name="time"):
        """Return fitted solutions as list of DataFrames.

        Use after run_fit. One DataFrame per dataset; failed integrations
        yield None at that index.

        Returns:
            list: Length = n_datasets. Each element is pd.DataFrame or None.

        Raises:
            RuntimeError: If run_fit not executed.
        """
        if self._param_info is None or self._result is None or self._fit_ctx is None:
            raise RuntimeError("run_fit not executed")

        solution_list = _integrate_datasets_for_params(
            self._fit_ctx, list(self._result.x)
        )
        names = self._param_info['function_names']
        result = []
        for sol in solution_list:
            if sol is None:
                result.append(None)
            else:
                result.append(_ode_result_to_dataframe(sol, names, time_column_name))
        return result


def run_fit_multi(builded_rxnode, df_list, p0, t_range=None,
                  method="RK45", rtol=1e-6, df_names=None,
                  opt_method='L-BFGS-B', bounds=None, verbose=True,
                  use_log_fit=False, lower_bound=None):
    """Convenience wrapper around ExpDataFitSci.run_fit.

    If t_range is None, derives from first DataFrame.
    use_log_fit and lower_bound are passed through to run_fit.

    Args:
        builded_rxnode (RxnODEbuild): Instance containing the reaction
            system definition.
        df_list (list[pandas.DataFrame]): List of experimental DataFrames.
            All must have same structure (time + species columns).
        p0 (list, tuple, or dict): Initial guess for symbolic rate constants.
            Same semantics as ExpDataFitSci.run_fit(p0=...): list/tuple in
            symbolic_rate_const_keys order, or dict with string keys (e.g.
            {"k1": 0.001, "k2": 0.002}).
        t_range (tuple[float, float], optional): Integration time span
            (t_start, t_end). If None, derives from first DataFrame.
            Defaults to None.
        df_names (list[str], optional): Names for each DataFrame. If None,
            uses df.attrs.get('name') when present; else str(i).
        method (str, optional): Integration method for solve_ivp.
            Defaults to "RK45".
        rtol (float, optional): Relative tolerance for solve_ivp.
            Defaults to 1e-6.
        opt_method (str, optional): scipy.optimize.minimize method.
            Defaults to 'L-BFGS-B'.
        bounds (list, optional): Bounds for each parameter (linear fit only).
            If None, uses lower_bound or default. Defaults to None.
        verbose (bool, optional): Print optimization result.
            Defaults to True.
        use_log_fit (bool, optional): If True, optimize in log(k) space.
            Passed to run_fit. Defaults to False.
        lower_bound (float, optional): Common lower bound for all parameters.
            Passed to run_fit. Defaults to None.

    Returns:
        tuple: (result, param_info, fit_metrics)
            - result: Object with .x (linear-scale k), .success, .fun,
              .tss, .r2. (RSS is .fun.)
            - param_info: Dict with symbolic_rate_consts, function_names,
              n_params, n_datasets, y0_list.
            - fit_metrics: Dict with keys 'rss', 'tss', 'r2'.

    Raises:
        ValueError: If df_list is empty, column structure is invalid,
            p0 has wrong length or invalid keys/values (see run_fit),
            lower_bound <= 0, or use_log_fit=True with non-positive p0.
    """
    if t_range is None:
        t_range = (
            float(df_list[0].iloc[:, 0].min()),
            float(df_list[0].iloc[:, 0].max())
        )
    fit_sci = ExpDataFitSci(
        builded_rxnode, df_list, t_range,
        method=method, rtol=rtol, df_names=df_names
    )
    return fit_sci.run_fit(
        p0,
        opt_method=opt_method,
        bounds=bounds,
        verbose=verbose,
        use_log_fit=use_log_fit,
        lower_bound=lower_bound,
    )

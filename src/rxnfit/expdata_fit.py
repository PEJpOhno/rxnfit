# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 01/03/2026, M. Ohno

"""Estimate unknown rate constants from experimental concentration trajectories.

Minimizers in ``scipy.optimize`` drive repeated ODE integrations through
:func:`rxnfit.solver_backend.solve_ode`; see there for LSODA / numbalsoda
behavior and SciPy fallbacks.
"""

import functools
import warnings
import numpy as np
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
from .solver_backend import solve_ode

from typing import Optional, List, Union

import pandas as pd


def _resolve_df_names(df_list, df_names=None):
    """Assign human-readable dataset labels.

    Explicit ``df_names`` win; otherwise each frame's ``attrs['name']`` or its
    index as a string is used.

    Args:
        df_list: Experimental tables in fit order.
        df_names: Optional parallel list matching ``df_list`` length.

    Returns:
        List of canonical names.

    Raises:
        ValueError: Length mismatch between ``df_list`` and ``df_names``.
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
    """Predict concentrations at observation times for one parameter vector.

    Args:
        t: Time samples (converted to sorted unique ``float64``).
        *params: Values for ``fit_ctx['symbolic_rate_const_keys']`` in order.
        fit_ctx: Bundle assembled by :func:`solve_fit_model` (RHS, spans, tolerances).

    Returns:
        Array shaped ``(n_species, n_times)``.

    Raises:
        ValueError: Missing parameters or invalid ``t``.
        RuntimeError: Integrator failure.
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

    time_dep = 'evaluator' in fit_ctx or has_time_dependent_rates(
        fit_ctx['builder'].rate_consts_dict
    )
    numb_ctx = {
        'builder': fit_ctx['builder'],
        'time_dependent': time_dep,
        'rate_const_values': rate_const_values,
        'symbolic_rate_const_keys': symbolic_rate_const_keys,
    }
    system_rhs = create_system_rhs(
        ode_functions_with_rate_consts,
        function_names,
        rate_const_values=rate_const_values,
        symbolic_rate_const_keys=symbolic_rate_const_keys,
        numbalsoda_context=numb_ctx,
    )

    if isinstance(t, (list, np.ndarray)):
        t_eval = np.array(t, dtype=np.float64)
    else:
        raise ValueError("t must be an array.")
    t_eval = np.sort(np.unique(t_eval.ravel()))

    try:
        solution = solve_ode(
            system_rhs,
            t_span,
            y0_fixed,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            time_dependent=time_dep,
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
    """Run one integration per dataset while sharing kinetic parameters ``params``.

    Args:
        fit_ctx: Structure from :func:`solve_fit_model_multi`.
        params: Candidate constants aligned with ``symbolic_rate_const_keys``.

    Returns:
        Parallel list of successful ``OdeResult`` instances or ``None`` slots.
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

    time_dep = 'evaluator' in fit_ctx or has_time_dependent_rates(
        fit_ctx['builder'].rate_consts_dict
    )
    numb_ctx = {
        'builder': fit_ctx['builder'],
        'time_dependent': time_dep,
        'rate_const_values': rate_const_values,
        'symbolic_rate_const_keys': symbolic_rate_const_keys,
    }
    system_rhs = create_system_rhs(
        ode_functions_with_rate_consts,
        function_names,
        rate_const_values=rate_const_values,
        symbolic_rate_const_keys=symbolic_rate_const_keys,
        numbalsoda_context=numb_ctx,
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
            solution = solve_ode(
                system_rhs,
                t_span_ds,
                y0,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                time_dependent=time_dep,
            )
            if not solution.success:
                solution_list.append(None)
            else:
                solution_list.append(solution)
        except Exception:
            solution_list.append(None)

    return solution_list


def _compute_multi_residual(params, fit_ctx):
    """Scalar objective: summed squared errors over every dataset/species/time.

    Args:
        params: Candidate constants in ``symbolic_rate_const_keys`` order.
        fit_ctx: Cached multi-dataset problem description.

    Returns:
        Finite loss or ``numpy.inf`` if any trajectory is missing.

    Note:
        Model concentrations are interpolated to experimental timestamps.
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
    """Build ``f(t, *params)`` that predicts species trajectories during fitting.

    Bare symbolic rate constants (``Symbol``) are optimized via ``params``; every
    other ``rate_consts_dict`` entry (numbers, expressions, etc.) stays in the
    merged constants dict passed to the RHS. ``fixed_initial_values`` are fixed ICs.

    Args:
        builded_rxnode: Configured reaction network.
        fixed_initial_values: Species IC vector aligned with ``function_names``.
        t_span: Global integration window covering every experimental time.
        method: Passed through to :func:`~rxnfit.solver_backend.solve_ode`.
        rtol: Relative tolerance for integrations.

    Returns:
        functools.partial wrapping :func:`_eval_ode_fit` plus a ``param_info``
        dict attached as attribute ``param_info``.

    Raises:
        ValueError: Wrong IC length versus species count.
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
        'builder': builded_rxnode,
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
    """Prepare multi-experiment residuals sharing one kinetic parameter vector.

    Each DataFrame yields its own ``(y0, t0)``; integrations stop at ``t_span[1]``.

    Args:
        builded_rxnode: Configured reaction network.
        df_list: Homogeneous experimental tables (time + species columns).
        t_span: Global window enclosing every observation time.
        method: Passed to :func:`~rxnfit.solver_backend.solve_ode`.
        rtol: Relative integration tolerance.
        df_names: Optional labels; see :func:`_resolve_df_names`.

    Returns:
        ``(residual_func, param_info, fit_ctx)`` where ``residual_func(params)``
        returns the summed squared error and ``fit_ctx`` backs plotting helpers.

    Raises:
        ValueError: Empty ``df_list`` or invalid column layout.
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
        'builder': builded_rxnode,
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
    """Normalize optimizer seeds to aligned float lists.

    Args:
        p0: Dict keyed by symbolic names or sequence in declaration order.
        param_info: Carries ``symbolic_rate_consts`` and ``n_params``.

    Returns:
        Ordered float list understood by SciPy wrappers.

    Raises:
        ValueError: Extra/missing dict keys, non-numeric entries, or bad length.
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


class ExpDataFit:
    """High-level façade over pooled least-squares fits and post-fit replay.

    Internally drives :func:`solve_fit_model_multi` and exposes helpers compatible
    with :class:`~rxnfit.solv_ode.RxnODEsolver`.
    """

    def __init__(self, builded_rxnode, df_list, t_range,
                 method="RK45", rtol=1e-6, df_names=None):
        """Cache builder, datasets, span, tolerances, and resolved names.

        Args:
            builded_rxnode: Reaction network scaffold.
            df_list: Parallel experimental traces.
            t_range: Outer integration bounds tuple.
            method: Passed to underlying ODE integrations.
            rtol: Relative solver tolerance.
            df_names: Optional labels for plotting; default resolution matches
                :meth:`plot_fitted_solution` expectations.
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
        """Optimize rate constants via :func:`scipy.optimize.minimize`.

        Args:
            p0: Sequence aligned with symbolic keys or explicit name→value mapping.
                Log-space mode rejects non-positive entries.
            opt_method: SciPy method string forwarded unchanged.
            bounds: Box constraints for linear mode; superseded under log-fit.
            verbose: Toggle progress printing.
            use_log_fit: Optimize ``log(k)`` internally while exposing linear ``k``.
            lower_bound: Optional shared positivity floor (mode dependent).

        Returns:
            ``(OptimizeResult-like, param_info, metrics_dict)``.

        Raises:
            ValueError: Invalid seeds, bounds, or incompatible log guesses.
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
                """Residual in log-parameter space with a finite fallback value.

                Args:
                    p: ``log(k)`` vector of same length as the rate-parameter vector.

                Returns:
                    Scalar residual, or ``1e15`` when the nested evaluation is non-finite.
                """
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
        """Build kwargs for :class:`~rxnfit.solv_ode.SolverConfig` post-fit.

        Args:
            dataset_index: Which pooled dataset supplies ``y0``/``t0``.

        Returns:
            Mapping with ``y0``, ``t_span``, integrator knobs, plus optional
            callable rate hooks for time-dependent kinetics.

        Raises:
            RuntimeError: If :meth:`run_fit` has not finished and stored state, or
                (when kinetics are time-dependent) ``_result`` is missing before
                building the evaluator.
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
        """Map symbolic names to optimized numeric rates.

        Args:
            result: External ``OptimizeResult`` or ``None`` to reuse cached output.

        Returns:
            Plain dict mergeable back into ``builded_rxnode.rate_consts_dict``.

        Raises:
            RuntimeError: Missing prior optimization state.
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
        """Replay best parameters with overlays mirroring ``plot_time_course_solutions``.

        Args:
            expdata_df: Optional replacement overlay tables; defaults to inputs.
            plot_datasets: Subset selector against ``df_names``.
            species: Restrict plotted columns; ``None`` keeps all.
            subplot_layout: Manual subplot grid sizing.

        Returns:
            ``None``

        Raises:
            RuntimeError: Missing fit artifacts.
            ValueError: Unknown dataset labels or dataframe shape mismatches.
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
        """Materialize every replayed trajectory via :func:`_ode_result_to_dataframe`.

        Args:
            time_column_name: Leading column label for timestamps.

        Returns:
            Parallel list mixing ``pandas.DataFrame`` entries and ``None`` holes.

        Raises:
            RuntimeError: Missing optimized state.
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
    """Stateful-free shortcut constructing :class:`ExpDataFit` then :meth:`~ExpDataFit.run_fit`.

    When ``t_range`` is omitted it becomes the min/max of the first dataframe's time column.

    Args:
        builded_rxnode: Reaction scaffold.
        df_list: Experimental bundle.
        p0: Seeds understood by :meth:`ExpDataFit.run_fit`.
        t_range: Optional manual span tuple.
        method, rtol, df_names: Forwarded to the constructor.
        opt_method, bounds, verbose, use_log_fit, lower_bound: Forwarded to :meth:`~ExpDataFit.run_fit`.

    Returns:
        Same triple as :meth:`ExpDataFit.run_fit`.

    Raises:
        ValueError: Validation errors bubbled up from pooling or SciPy phases.
    """
    if t_range is None:
        t_range = (
            float(df_list[0].iloc[:, 0].min()),
            float(df_list[0].iloc[:, 0].max())
        )
    fit_sci = ExpDataFit(
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

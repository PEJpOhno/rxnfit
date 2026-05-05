# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

"""Optuna search for initial rate parameters before gradient-based fitting.

:class:`P0OptFit` proposes trial points with Optuna, scores them with a light
wrapper around :class:`rxnfit.expdata_fit.ExpDataFit`, and returns refined
estimates for downstream ``scipy.optimize`` runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import optuna
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

from .build_ode import RxnODEbuild
from .expdata_fit import ExpDataFit

# Default exploration range (low, high) per spec
_DEFAULT_LOW = 1e-8
_DEFAULT_HIGH = 1e2

# Default exceptions to catch so optimization continues after a trial failure
_DEFAULT_CATCH: Tuple[Type[BaseException], ...] = (TrialPruned, RuntimeError, ValueError)


def _default_t_range_from_dfs(df_list: list) -> Tuple[float, float]:
    """Compute t_range as (0, max of first column across all DataFrames)."""
    if not df_list:
        raise ValueError("df_list cannot be empty.")
    max_t = max(float(df.iloc[:, 0].max()) for df in df_list)
    return (0.0, max_t)


def _resolve_param_bounds(
    param_bounds: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]],
    symbolic_keys: List[str],
) -> List[Tuple[float, float]]:
    """Resolve param_bounds to a list of (low, high) in symbolic_keys order.

    - If param_bounds is None or (low, high): use (1e-8, 1e2) for all.
    - If dict: each key must be in symbolic_keys (else ValueError).
      Missing keys get (1e-8, 1e2).
    """
    default = (_DEFAULT_LOW, _DEFAULT_HIGH)
    if param_bounds is None:
        return [default] * len(symbolic_keys)
    if isinstance(param_bounds, (list, tuple)) and len(param_bounds) == 2:
        low, high = float(param_bounds[0]), float(param_bounds[1])
        return [(low, high)] * len(symbolic_keys)
    if isinstance(param_bounds, dict):
        allowed = set(symbolic_keys)
        for key in param_bounds:
            if key not in allowed:
                raise ValueError(
                    f"param_bounds contains a key that is not a symbolic rate constant: {key!r}. "
                    f"Valid keys: {sorted(allowed)}."
                )
        return [
            param_bounds.get(name, default)
            for name in symbolic_keys
        ]
    raise TypeError(
        "param_bounds must be a (low, high) tuple or a dict with variable names as keys."
    )


class P0OptFit:
    """Optimize initial parameter values (p0) for ODE rate constants using Optuna, then fit with ExpDataFit.

    By default no random seed is set; Optuna uses its default (TPESampler without seed).
    Pass seed in the constructor for reproducible search.
    """

    def __init__(
        self,
        reaction_source: Union[str, RxnODEbuild],
        df_list: list,
        t_range: Optional[Tuple[float, float]] = None,
        encoding: str = "utf-8",
        param_bounds: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
        df_names: Optional[List[str]] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        opt_method: str = "L-BFGS-B",
        use_log_fit: bool = False,
        lower_bound: Optional[float] = None,
        verbose: bool = True,
        storage: Optional[Union[str, optuna.storages.BaseStorage]] = None,
        seed: Optional[int] = None,
        catch: Optional[Union[Tuple[Type[BaseException], ...], List[Type[BaseException]]]] = None,
    ):
        """Initialize the p0 optimizer.

        Args:
            reaction_source: Path to reaction CSV (str) or existing RxnODEbuild instance.
            df_list: List of experimental DataFrames (time + species columns).
            t_range: Integration time span (t_start, t_end). If None, (0, max of first column).
            encoding: Encoding for reaction CSV when reaction_source is a path. Default utf-8.
            param_bounds: (low, high) for all params, or dict {name: (low, high)}. Default (1e-8, 1e2).
            df_names: Names for each DataFrame (for plot_fitted_solution).
                If None, uses df.attrs.get('name') when present; else str(i).
            method: Integration method for solve_ivp. Default "RK45".
            rtol: Relative tolerance for solve_ivp. Default 1e-6.
            opt_method: scipy.optimize.minimize method for run_fit. Default "L-BFGS-B".
            use_log_fit: If True, optimize in log space. Default False.
            lower_bound: Common lower bound for parameters (run_fit). Default None.
            verbose: Whether to print run_fit result. Default True.
            storage: Optuna storage (e.g. RDB URL). None for in-memory.
            seed: Random seed for Optuna sampler. If None (default), no seed is set;
                Optuna uses its default (sampler not specified, so TPESampler is used).
                If given, TPESampler(seed=seed) is used for reproducible search.
            catch: Exception types to catch so that a failed trial does not stop optimization.
                Tuple or list of exception classes. If None, uses (TrialPruned, RuntimeError,
                ValueError). If empty tuple/list, no exceptions are caught (strict mode;
                any exception in the objective will stop optimization).
        """
        if not df_list:
            raise ValueError("df_list cannot be empty.")

        if isinstance(reaction_source, RxnODEbuild):
            self._builded_rxnode = reaction_source
        else:
            self._builded_rxnode = RxnODEbuild(reaction_source, encoding=encoding)

        self._df_list = df_list
        if t_range is None:
            t_range = _default_t_range_from_dfs(df_list)
        self._t_range = t_range

        symbolic_keys = self._builded_rxnode.get_symbolic_rate_const_keys()
        if not symbolic_keys:
            raise ValueError(
                "No symbolic rate constants. Nothing to fit."
            )
        self._symbolic_keys = symbolic_keys
        self._bounds_per_param = _resolve_param_bounds(param_bounds, symbolic_keys)
        self._df_names = df_names

        self._method = method
        self._rtol = rtol
        self._opt_method = opt_method
        self._use_log_fit = use_log_fit
        self._lower_bound = lower_bound
        self._verbose = verbose
        self._storage = storage
        self._seed = seed
        if catch is None:
            self._catch: Tuple[Type[BaseException], ...] = _DEFAULT_CATCH
        else:
            self._catch = tuple(catch)

        self._study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: suggest p0, run run_fit, return residual."""
        p0 = []
        for i, name in enumerate(self._symbolic_keys):
            low, high = self._bounds_per_param[i]
            val = trial.suggest_float(name, low, high, log=True)
            p0.append(val)

        bounds = [(self._bounds_per_param[i][0], None) for i in range(len(self._symbolic_keys))]

        fit = ExpDataFit(
            self._builded_rxnode,
            self._df_list,
            self._t_range,
            method=self._method,
            rtol=self._rtol,
            df_names=self._df_names,
        )
        result, _, fit_metrics_ret = fit.run_fit(
            p0,
            opt_method=self._opt_method,
            bounds=bounds,
            verbose=False,
            use_log_fit=self._use_log_fit,
            lower_bound=self._lower_bound,
        )
        if not result.success:
            raise RuntimeError(
                f"run_fit did not converge: {getattr(result, 'message', '')}"
            )
        return float(result.fun)

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """Run Optuna optimization and return best p0 and fitted result.

        Args:
            n_trials: Number of trials. Passed to study.optimize().
            timeout: Timeout in seconds. Passed to study.optimize().
            n_jobs: Number of parallel jobs. Passed to study.optimize().
            show_progress_bar: Whether to show progress bar. Default True.
            **kwargs: Additional arguments passed to study.optimize().
                Note: 'catch' is ignored; use the constructor argument to set it.

        Returns:
            Tuple of (dict, fit_metrics):
                - dict: { variable_name: (optimal_initial_value, fitted_value) }
                - fit_metrics: Dict with keys 'rss', 'tss', 'r2'.

        Raises:
            RuntimeError: If all trials fail.
        """
        kwargs_opt = {k: v for k, v in kwargs.items() if k != "catch"}
        if self._seed is not None:
            sampler = optuna.samplers.TPESampler(seed=self._seed)
            study = optuna.create_study(direction="minimize", storage=self._storage, sampler=sampler)
        else:
            study = optuna.create_study(direction="minimize", storage=self._storage)
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            catch=self._catch,
            **kwargs_opt,
        )
        self._study = study

        complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("All trials failed.")

        best = min(complete, key=lambda t: t.value)
        p0_best = [best.params[name] for name in self._symbolic_keys]
        bounds = [(self._bounds_per_param[i][0], None) for i in range(len(self._symbolic_keys))]

        fit = ExpDataFit(
            self._builded_rxnode,
            self._df_list,
            self._t_range,
            method=self._method,
            rtol=self._rtol,
            df_names=self._df_names,
        )
        result, _, fit_metrics_ret = fit.run_fit(
            p0_best,
            opt_method=self._opt_method,
            bounds=bounds,
            verbose=False,
            use_log_fit=self._use_log_fit,
            lower_bound=self._lower_bound,
        )
        out_dict = {
            name: (p0_best[i], float(result.x[i]))
            for i, name in enumerate(self._symbolic_keys)
        }
        if self._verbose:
            print(f"Optimization success: {result.success}")
            print("Fitted rate constants:")
            for i, name in enumerate(self._symbolic_keys):
                print(f"  {name} = {result.x[i]:.6g}")
            print(
                f"Residual sum of squares: {fit_metrics_ret['rss']:.6g}  "
                f"R²: {fit_metrics_ret['r2']:.6g}"
            )
        return (out_dict, fit_metrics_ret)

    def optuna_log(self) -> List[Dict[str, Any]]:
        """Return per-trial log. Empty list if optimize() has not been run.

        Returns:
            List of dicts with keys: trial_No, params, rss, state.
        """
        if self._study is None:
            return []
        logs = []
        for i, trial in enumerate(self._study.trials):
            logs.append({
                "trial_No": i + 1,
                "params": dict(trial.params),
                "rss": trial.value,
                "state": trial.state.name,
            })
        return logs

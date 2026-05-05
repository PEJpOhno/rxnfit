# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 09/07/2025, M. Ohno

"""Integrate and plot concentration trajectories when rate laws are fully specified.

Uses :func:`rxnfit.solver_backend.solve_ode` for time stepping (including
optional LSODA / numbalsoda; see that module for dispatch rules and fallbacks).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Callable, Dict, Tuple
import sys
import warnings
from sympy import Basic as SympyBasic

import numpy as np
import pandas as pd

from .build_ode import RxnODEbuild, create_system_rhs
from .rate_const_ft_eval import has_time_dependent_rates
from .fit_metrics import fit_metrics as compute_fit_metrics
from .fit_metrics import expdata_df_to_datasets, TSS_MIN_THRESHOLD
from .plot_results import plot_time_course_solutions
from .solver_backend import (
    LsodaImplicitTEvalMode,
    build_t_eval_when_none_for_lsoda,
    solve_ode,
)


def _ode_result_to_dataframe(sol, names, time_column_name="time"):
    """Convert an ODE solution object to a single time-course DataFrame.

    Args:
        sol: Object with ``t`` and ``y`` (SciPy ``OdeResult`` or compatible).
        names: Species column names in the same order as ``y`` rows.
        time_column_name: Name of the time column. Defaults to ``"time"``.

    Returns:
        DataFrame whose first column is time and remaining columns are species.
    """
    data = {time_column_name: sol.t}
    for i, name in enumerate(names):
        data[name] = sol.y[i]
    return pd.DataFrame(data)


@dataclass
class SolverConfig:
    """User-facing options for :class:`RxnODEsolver`.

    Attributes:
        y0: Initial state (species concentrations in ``function_names`` order).
        t_span: Integration window ``(t_start, t_end)``.
        t_eval: Optional dense output times. For ``method="LSODA"`` without
            time-dependent rates, a grid may be built automatically when omitted.
        method: Integrator label passed to :func:`~rxnfit.solver_backend.solve_ode`.
        rtol: Relative tolerance.
        rate_const_values: Fixed dict ``k -> float`` or ``(t) -> dict`` when
            using lambdified ODEs with explicit rate arguments.
        symbolic_rate_const_keys: Parameter order for those ODE signatures.
            ``rate_const_values`` and ``symbolic_rate_const_keys`` must both be set
            or both omitted.
    """
    y0: list
    t_span: tuple
    t_eval: Optional[np.ndarray] = field(default=None)
    method: str = "RK45"
    rtol: float = 1e-6
    rate_const_values: Optional[Union[Dict[str, float], Callable[[float], Dict[str, float]]]] = field(default=None)
    symbolic_rate_const_keys: Optional[List[str]] = field(default=None)


def _solver_time_dependent(builder: RxnODEbuild, config: SolverConfig) -> bool:
    rcv = getattr(config, "rate_const_values", None)
    if callable(rcv):
        return True
    return has_time_dependent_rates(builder.rate_consts_dict)


class RxnODEsolver:
    """Integrate reaction-network ODEs and compare against experiments.

    Time stepping is delegated to :func:`~rxnfit.solver_backend.solve_ode`
    (LSODA / numbalsoda policy is documented there). Plotting and RSS-based
    metrics reuse :mod:`rxnfit.plot_results` and :mod:`rxnfit.fit_metrics`.

    Attributes:
        builder: Reaction system and RHS factory.
        config: Solver options.
        ode_construct: Tuple from the builder after :meth:`solve_system`.
        solution: Latest integration result, or ``None`` if integration failed.
    """

    def __init__(self, builder: RxnODEbuild, config: SolverConfig):
        """Validate configuration and attach the builder.

        Args:
            builder: Prepared :class:`~rxnfit.build_ode.RxnODEbuild` instance.
            config: Non-conflicting :class:`SolverConfig`.

        Raises:
            ValueError: If only one of ``rate_const_values`` and
                ``symbolic_rate_const_keys`` is set.
            SystemExit: Process exits with status 1 when stored rate constants
                are neither numeric nor SymPy-expressible (legacy behavior).
        """
        # rate_const_values と symbolic_rate_const_keys は両方指定か両方省略
        rcv = getattr(config, 'rate_const_values', None)
        srck = getattr(config, 'symbolic_rate_const_keys', None)
        if (rcv is None) != (srck is None):
            raise ValueError(
                "rate_const_values and symbolic_rate_const_keys must be "
                "both set or both omitted; one alone is invalid."
            )

        # 反応速度定数の型をチェック
        if not self._validate_rate_constants(builder.rate_consts_dict):
            print("reconfirm rate constants")
            sys.exit(1)
        
        # ODEビルダーのインスタンスを受け取り、参照として保持
        self.builder = builder
        self.config = config
        self.ode_construct = None
        self.solution = None

    def _validate_rate_constants(self, rate_consts_dict):
        """Return whether every rate entry is a Python ``int``/``float`` or SymPy ``Basic``.

        Args:
            rate_consts_dict: Mapping of rate names to values (must be a dict).

        Returns:
            ``True`` only when all values pass the check (``False`` if not a dict).
        """
        if not isinstance(rate_consts_dict, dict):
            return False

        for key, value in rate_consts_dict.items():
            # Accept Python numeric types, NumPy numeric types,
            # and SymPy symbolic values
            if isinstance(value, SympyBasic):
                continue
            if isinstance(value, (int, float)):
                continue
            return False

        return True

    def solve_system(self):
        """Integrate the model and cache ``ode_construct`` and ``solution``.

        Returns:
            A pair ``(ode_construct, solution)``. ``solution`` is ``None`` only
            when integration raises; otherwise it is the integrator output (check
            ``success`` on SciPy-style results). Failures without exceptions still
            return a result object.
        """
        rcv = getattr(self.config, 'rate_const_values', None)
        srck = getattr(self.config, 'symbolic_rate_const_keys', None)
        td = _solver_time_dependent(self.builder, self.config)
        method_u = str(self.config.method).upper()
        t_eval = self.config.t_eval
        if method_u == "LSODA" and t_eval is None and not td:
            t_eval = build_t_eval_when_none_for_lsoda(
                LsodaImplicitTEvalMode.SOLVE_SYSTEM,
                t_span=self.config.t_span,
            )
        numb_ctx = {
            "builder": self.builder,
            "time_dependent": td,
            "rate_const_values": rcv,
            "symbolic_rate_const_keys": srck,
        }

        if rcv is not None and srck is not None:
            ode_system, symbolic_rate_const_keys = (
                self.builder.create_ode_system_with_rate_consts()
            )
            system_rhs = create_system_rhs(
                ode_system,
                self.builder.function_names,
                rate_const_values=rcv,
                symbolic_rate_const_keys=srck,
                numbalsoda_context=numb_ctx,
            )
        else:
            ode_construct = self.builder.get_ode_system()
            (system_of_equations, sympy_symbol_dict,
             ode_system, function_names, rate_consts_dict) = ode_construct
            system_rhs = create_system_rhs(
                ode_system,
                function_names,
                numbalsoda_context=numb_ctx,
            )

        solution = None
        try:
            if t_eval is not None:
                t_eval = np.sort(np.unique(np.asarray(t_eval, dtype=np.float64).ravel()))
            solution = solve_ode(
                system_rhs,
                self.config.t_span,
                self.config.y0,
                t_eval=t_eval,
                method=self.config.method,
                rtol=self.config.rtol,
                time_dependent=td,
            )
        except Exception as e:
            print(f"An error occurred during numerical integration.: {e}")
            print("Plz. review the debug info.")

        if rcv is not None and srck is not None:
            ode_construct = (
                self.builder.get_equations(),
                self.builder.sympy_symbol_dict,
                ode_system,
                self.builder.function_names,
                self.builder.rate_consts_dict,
            )
        else:
            ode_construct = self.builder.get_ode_system()
        self.ode_construct = ode_construct
        self.solution = solution
        return ode_construct, solution

    def to_dataframe_list(self, solution=None, time_column_name="time"):
        """Convert ``solution`` (or the cached one) to a one-element DataFrame list.

        Args:
            solution: Optional override; defaults to :attr:`solution`.
            time_column_name: Name of the leading time column.

        Returns:
            ``[DataFrame]`` with species columns named like ``function_names``.

        Raises:
            RuntimeError: When no solution exists.
        """
        sol = solution if solution is not None else self.solution
        if sol is None:
            raise RuntimeError(
                "No solution available. Call solve_system() first."
            )
        names = self.builder.function_names
        return [_ode_result_to_dataframe(sol, names, time_column_name)]

    def _compute_rss(self, expdata_df, solution=None, recompute=True):
        """Residual sum of squares against ``expdata_df``.

        Args:
            expdata_df: Experimental table; column 0 is time, others species.
            solution: Model trajectory used when ``recompute`` is False.
            recompute: If True, integrate again on the union of experimental times.

        Returns:
            Scalar RSS.

        Raises:
            RuntimeError: Missing ``ode_construct``, no valid experimental times,
                or trajectory missing when interpolation is needed.

        Note:
            If re-integration raises or returns ``success=False``, a warning is
            issued and RSS falls back to :meth:`_compute_rss_interp` instead of
            raising immediately.
        """
        if recompute:
            if self.ode_construct is None:
                raise RuntimeError(
                    "recompute=True requires solve_system() to be called first."
                )
            rcv = getattr(self.config, 'rate_const_values', None)
            srck = getattr(self.config, 'symbolic_rate_const_keys', None)
            td = _solver_time_dependent(self.builder, self.config)
            numb_ctx = {
                "builder": self.builder,
                "time_dependent": td,
                "rate_const_values": rcv,
                "symbolic_rate_const_keys": srck,
            }
            if rcv is not None and srck is not None:
                ode_system, _ = self.builder.create_ode_system_with_rate_consts()
                system_rhs = create_system_rhs(
                    ode_system,
                    self.builder.function_names,
                    rate_const_values=rcv,
                    symbolic_rate_const_keys=srck,
                    numbalsoda_context=numb_ctx,
                )
            else:
                (
                    _,
                    _,
                    ode_system,
                    function_names,
                    _,
                ) = self.ode_construct
                system_rhs = create_system_rhs(
                    ode_system,
                    function_names,
                    numbalsoda_context=numb_ctx,
                )
            t_col = expdata_df.iloc[:, 0].dropna()
            if len(t_col) == 0:
                raise RuntimeError(
                    "No valid time points in the experimental data."
                )
            t_eval = np.sort(np.unique(t_col.to_numpy()))
            try:
                sol_new = solve_ode(
                    system_rhs,
                    self.config.t_span,
                    self.config.y0,
                    t_eval=t_eval,
                    method=self.config.method,
                    rtol=self.config.rtol,
                    time_dependent=td,
                )
                if not sol_new.success:
                    raise RuntimeError(
                        sol_new.message or "Integration failed."
                    )
            except Exception as e:
                warnings.warn(
                    f"Re-integration at experimental times failed ({e}). "
                    "Computing residual by interpolation.",
                    UserWarning,
                    stacklevel=2,
                )
                return self._compute_rss_interp(expdata_df, solution)
            time_to_idx = {float(t): idx for idx, t in enumerate(sol_new.t)}
            names = self.builder.function_names
            name_to_idx = {name: i for i, name in enumerate(names)}
            rss = 0.0
            for col in expdata_df.columns[1:]:
                if col not in name_to_idx:
                    continue
                i = name_to_idx[col]
                c_exp = expdata_df[col].to_numpy()
                for j in range(len(expdata_df)):
                    t_j = expdata_df.iloc[j, 0]
                    if pd.isna(t_j) or np.isnan(c_exp[j]):
                        continue
                    t_j = float(t_j)
                    idx = time_to_idx[t_j]
                    rss += (c_exp[j] - sol_new.y[i][idx]) ** 2
            return float(rss)

        return self._compute_rss_interp(expdata_df, solution)

    def _compute_rss_interp(self, expdata_df, solution=None):
        """RSS via linear interpolation of an existing trajectory.

        Args:
            expdata_df: Same layout as :meth:`_compute_rss`.
            solution: Trajectory to interpolate; defaults to :attr:`solution`.

        Returns:
            Scalar RSS.

        Raises:
            RuntimeError: When no trajectory is available.
        """
        sol = solution if solution is not None else self.solution
        if sol is None:
            raise RuntimeError(
                "No solution available. Call solve_system() first."
            )
        t_exp = expdata_df.iloc[:, 0].to_numpy()
        names = self.builder.function_names
        name_to_idx = {name: i for i, name in enumerate(names)}
        rss = 0.0
        for col in expdata_df.columns[1:]:
            if col not in name_to_idx:
                continue
            i = name_to_idx[col]
            c_exp = expdata_df[col].to_numpy()
            mask = ~np.isnan(c_exp)
            if not np.any(mask):
                continue
            t_m = t_exp[mask]
            c_m = c_exp[mask]
            c_model = np.interp(t_m, sol.t, sol.y[i])
            rss += np.sum((c_m - c_model) ** 2)
        return float(rss)

    def eval_fit_metrics(self, expdata_df, solution=None, verbose=True, recompute=True):
        """Return ``{'rss', 'tss', 'r2'}`` using :func:`~rxnfit.fit_metrics.fit_metrics`.

        NaNs are dropped consistently; TSS is per-species about the experimental mean.

        Args:
            expdata_df: Time in column 0; species columns match ``function_names``.
            solution: Optional trajectory for the interpolation path.
            verbose: Print RSS and R² when true.
            recompute: Forwarded to :meth:`_compute_rss`.

        Returns:
            Metric dictionary of floats.

        Raises:
            RuntimeError: Same as :meth:`_compute_rss` (interpolation path).

        Note:
            :func:`~rxnfit.fit_metrics.expdata_df_to_datasets` may raise
            ``ValueError`` if species columns required by ``function_names`` are
            missing from ``expdata_df``.
        """
        rss = self._compute_rss(expdata_df, solution, recompute)
        datasets = expdata_df_to_datasets(expdata_df, self.builder.function_names)
        metrics = compute_fit_metrics(datasets, rss)
        if metrics["tss"] < TSS_MIN_THRESHOLD:
            warnings.warn(
                "TSS is nearly zero; R² may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        if verbose:
            print(
                f"Residual sum of squares: {metrics['rss']:.6g}  "
                f"R²: {metrics['r2']:.6g}"
            )
        return metrics

    def solution_plot(
        self,
        solution=None,
        expdata_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        species: Optional[List[str]] = None,
        subplot_layout: Optional[Tuple[int, int]] = None,
    ):
        """Plot concentrations and optional experimental overlays.

        Delegates layout to :func:`~rxnfit.plot_results.plot_time_course_solutions`.

        Args:
            solution: Trajectory to plot; defaults to :attr:`solution`.
            expdata_df: One or more DataFrames (time + species); multiple frames
                create one subplot each while reusing the same simulated curve.
            species: Subset of ``function_names``; ``None`` plots every species.
            subplot_layout: ``(rows, cols)`` when several DataFrames are supplied;
                ``None`` picks a vertical stack.

        Returns:
            ``None``. The plotting helper returns ``(fig, axes)``, which this
            method does not pass through.

        Raises:
            ValueError: Passed through when ``species`` names are absent from the
                ODE species list (see :func:`~rxnfit.plot_results.plot_time_course_solutions`).
        """
        sol = solution if solution is not None else self.solution
        if sol is None:
            print("No solution to plot. Run solve_system() first "
                  "or pass a solution.")
            return

        if expdata_df is not None:
            df_list = expdata_df if isinstance(expdata_df, list) else [expdata_df]
        else:
            df_list = []

        solution_list = [sol] * max(1, len(df_list))
        plot_time_course_solutions(
            solution_list,
            df_list,
            self.builder.function_names,
            species=species,
            subplot_layout=subplot_layout,
        )

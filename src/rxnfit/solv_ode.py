# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 09/07/2025, M. Ohno

"""
Calculate and visualize the time evolution of chemical species
based on reaction rate equations with all rate constants known.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Callable, Dict
import sys
import warnings
from sympy import Basic as SympyBasic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from .build_ode import RxnODEbuild, create_system_rhs
from .expdata_reader import get_time_unit_from_expdata


@dataclass
class SolverConfig:
    """Configuration parameters for the ODE solver.

    Attributes:
        y0 (list): Initial concentrations for all species (required).
        t_span (tuple): Time span for integration as (t_start, t_end)
            (required).
        t_eval (Optional[np.ndarray]): Time points at which to evaluate
            the solution. If None, the solver chooses the time points.
        method (str): Integration method to use. Defaults to "RK45".
        rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        rate_const_values (Optional[Union[dict, Callable[[float], dict]]]):
            When provided with symbolic_rate_const_keys, the solver uses the
            rate-constants ODE path: either a dict of rate constant values, or
            a callable (t) -> dict for time-dependent rates. Must be used
            together with symbolic_rate_const_keys (both or neither).
        symbolic_rate_const_keys (Optional[List[str]]): Order of rate constant
            names passed to the ODE. Required when rate_const_values is set.
            Both must be set or both omitted; otherwise an error is raised.
    """
    y0: list
    t_span: tuple
    t_eval: Optional[np.ndarray] = field(default=None)
    method: str = "RK45"
    rtol: float = 1e-6
    rate_const_values: Optional[Union[Dict[str, float], Callable[[float], Dict[str, float]]]] = field(default=None)
    symbolic_rate_const_keys: Optional[List[str]] = field(default=None)


class RxnODEsolver:
    """Solver for ODE systems representing chemical reaction kinetics.
    
    This class integrates ODE systems using scipy.solve_ivp and provides
    methods to visualize the time evolution of chemical species.
    
    Attributes:
        builder (RxnODEbuild): The ODE builder instance containing the
            reaction system definition.
        config (SolverConfig): Configuration parameters for the solver.
        ode_construct (tuple, optional): ODE system construction data.
            Set after calling solve_system().
        solution (scipy.integrate.OdeResult, optional): Solution from
            numerical integration. Set after calling solve_system().
    """
    
    def __init__(self, builder: RxnODEbuild, config: SolverConfig):
        """Initialize the ODE solver.
        
        Args:
            builder (RxnODEbuild): The ODE builder instance containing
                the reaction system definition.
            config (SolverConfig): Configuration parameters for the solver.
        
        Raises:
            SystemExit: If rate constants validation fails.
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
        """Validate that all rate constants are numeric or symbolic.
        
        Checks that all values in the rate constants dictionary are either
        numeric types (int, float) or SymPy symbolic expressions. This
        ensures the rate constants are suitable for numerical integration.
        
        Args:
            rate_consts_dict (dict): Dictionary of rate constants.
            
        Returns:
            bool: True if all values are numeric or symbolic, False otherwise.
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
        """Solve the ODE system numerically.

        Performs numerical integration of the ODE system using scipy.solve_ivp.
        The solution is stored internally and also returned.
        When config has rate_const_values and symbolic_rate_const_keys set,
        uses the rate-constants ODE path; otherwise uses the standard path.

        Returns:
            tuple: A tuple containing:
                - ode_construct (tuple): ODE system construction data including
                    system_of_equations, sympy_symbol_dict, ode_system,
                    function_names, and rate_consts_dict.
                - solution (scipy.integrate.OdeResult): Solution object from
                    scipy.solve_ivp containing time points and concentrations.
                    May be None if integration fails.
        """
        rcv = getattr(self.config, 'rate_const_values', None)
        srck = getattr(self.config, 'symbolic_rate_const_keys', None)

        if rcv is not None and srck is not None:
            ode_system, symbolic_rate_const_keys = (
                self.builder.create_ode_system_with_rate_consts()
            )
            system_rhs = create_system_rhs(
                ode_system,
                self.builder.function_names,
                rate_const_values=rcv,
                symbolic_rate_const_keys=srck,
            )
        else:
            ode_construct = self.builder.get_ode_system()
            (system_of_equations, sympy_symbol_dict,
             ode_system, function_names, rate_consts_dict) = ode_construct
            system_rhs = create_system_rhs(ode_system, function_names)

        solution = None
        try:
            solution = solve_ivp(
                system_rhs,
                self.config.t_span,
                self.config.y0,
                t_eval=self.config.t_eval,
                method=self.config.method,
                rtol=self.config.rtol
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

    def to_dataframe(self, solution=None, time_column_name="time"):
        """Return the solution as a DataFrame (time as rows, species as columns).

        Args:
            solution (scipy.integrate.OdeResult, optional): Solution to
                convert. If None, uses the solution from solve_system().
                Defaults to None.
            time_column_name (str, optional): Name of the time column
                (first column). Defaults to "time".

        Returns:
            pandas.DataFrame: One row per time point; first column is time,
                following columns are species concentrations (names from
                function_names).

        Raises:
            RuntimeError: If no solution is available (solve_system not run).
        """
        sol = solution if solution is not None else self.solution
        if sol is None:
            raise RuntimeError(
                "No solution available. Call solve_system() first."
            )
        names = self.builder.function_names
        data = {time_column_name: sol.t}
        for i, name in enumerate(names):
            data[name] = sol.y[i]
        return pd.DataFrame(data)

    def rsq(self, expdata_df, solution=None, verbose=True, recompute=True):
        """Compute the sum of squared residuals at experimental time points.

        When recompute=True (default), re-integrates the ODE at the valid
        experimental time points and returns the sum of (observed - model)^2.
        When recompute=False, interpolates the existing solution at
        experimental times to compute the residual. NaNs in the data are
        excluded. Output format matches expdata_fit_sci.run_fit.

        Args:
            expdata_df (pandas.DataFrame): Time-course data. First column
                is time; remaining columns are species concentrations.
                Column names must match function_names.
            solution (scipy.integrate.OdeResult, optional): Solution to use
                when recompute=False. If None, uses the result of
                solve_system(). Defaults to None.
            verbose (bool, optional): If True, print the residual sum of
                squares. Defaults to True.
            recompute (bool, optional): If True, re-integrate at
                experimental times to compute residuals. If False, use
                interpolation. Defaults to True.

        Returns:
            float: Sum of squared residuals.

        Raises:
            RuntimeError: If no solution is available, or if recompute=True
                but solve_system() has not been called.
        """
        if recompute:
            if self.ode_construct is None:
                raise RuntimeError(
                    "recompute=True requires solve_system() to be called first."
                )
            rcv = getattr(self.config, 'rate_const_values', None)
            srck = getattr(self.config, 'symbolic_rate_const_keys', None)
            if rcv is not None and srck is not None:
                ode_system, _ = self.builder.create_ode_system_with_rate_consts()
                system_rhs = create_system_rhs(
                    ode_system,
                    self.builder.function_names,
                    rate_const_values=rcv,
                    symbolic_rate_const_keys=srck,
                )
            else:
                (
                    _,
                    _,
                    ode_system,
                    function_names,
                    _,
                ) = self.ode_construct
                system_rhs = create_system_rhs(ode_system, function_names)
            t_col = expdata_df.iloc[:, 0].dropna()
            if len(t_col) == 0:
                raise RuntimeError(
                    "No valid time points in the experimental data."
                )
            t_eval = np.sort(np.unique(t_col.to_numpy()))
            try:
                sol_new = solve_ivp(
                    system_rhs,
                    self.config.t_span,
                    self.config.y0,
                    t_eval=t_eval,
                    method=self.config.method,
                    rtol=self.config.rtol,
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
                return self._rsq_interp(expdata_df, solution, verbose)
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
            if verbose:
                print(f"Residual sum of squares: {rss:.6g}")
            return float(rss)

        return self._rsq_interp(expdata_df, solution, verbose)

    def _rsq_interp(self, expdata_df, solution=None, verbose=True):
        """Compute sum of squared residuals by interpolating the solution.

        Used when rsq is called with recompute=False.
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
        if verbose:
            print(f"Residual sum of squares: {rss:.6g}")
        return float(rss)

    # Plot results
    def solution_plot(
        self,
        solution=None,
        expdata_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        species: Optional[List[str]] = None,
    ):
        """Plot the time evolution of chemical species.
        
        Creates a time-course plot showing the simulated concentration of
        each species over time. Optionally overlays experimental data
        points with colors matching the simulation lines.
        Also prints the final concentrations at the last time point.

        Args:
            solution (scipy.integrate.OdeResult, optional): Solution
                object from solve_ivp to plot. If None, uses the solution
                stored internally from solve_system(). Defaults to None.
            expdata_df (pandas.DataFrame or list of DataFrame, optional):
                Experimental data for overlay. Can be a single DataFrame or
                a list of DataFrames (e.g. from multiple CSVs). Format:
                - First column: time values (column name used for x-axis
                  label unit, e.g. "t_s" -> "Time (s)").
                - Subsequent columns: concentrations for each species.
                - Column names must match species names (e.g. from
                  self.builder.function_names).
                If a list is passed, the 0th column name must be identical
                across all DataFrames; otherwise a warning is emitted.
                Scatter points and x-axis range/label are derived from
                the first DataFrame when a list is given. Defaults to None.
            species (list[str], optional): List of species names to plot.
                If None, all species in the ODE system are plotted.
                If provided, only these species are drawn. Any name not
                in the ODE system triggers a warning and ValueError.
                Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If a name in species is not in the ODE system.
        """
        sol = solution if solution is not None else self.solution
        if sol is None:
            print("No solution to plot. Run solve_system() first "
                  "or pass a solution.")
            return

        all_species = self.builder.function_names
        if species is None:
            plot_species = list(all_species)
        else:
            invalid = [s for s in species if s not in all_species]
            if invalid:
                warnings.warn(
                    f"Species not in the ODE system: {invalid}. "
                    f"Available: {all_species}. Please check the argument.",
                    UserWarning,
                    stacklevel=2,
                )
                raise ValueError(
                    f"Species not in the ODE system: {invalid}. "
                    f"Available: {all_species}"
                )
            plot_species = list(species)

        # プロットする (名前, 解のインデックス) のリスト
        name_to_idx = {name: i for i, name in enumerate(all_species)}
        plot_items = [(name, name_to_idx[name]) for name in plot_species]

        # expdata_df を正規化し、単位は expdata_reader で取得・不一致時はそこで警告
        plot_df = None
        time_unit = None
        if expdata_df is not None:
            df_list = expdata_df if isinstance(expdata_df, list) else [expdata_df]
            plot_df = df_list[0]
            time_unit = get_time_unit_from_expdata(df_list)

        print("\n=== Time-course plot ===")
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # 解の時間範囲（実験データあり時に軸をここに合わせ、実線が圧縮されないようにする）
        t_sol_min, t_sol_max = float(sol.t.min()), float(sol.t.max())

        for species_name, i in plot_items:
            line, = ax.plot(sol.t, sol.y[i], label=species_name, linewidth=2)
            color = line.get_color()
            if plot_df is not None and species_name in plot_df.columns:
                t_exp = plot_df.iloc[:, 0].to_numpy()
                c_exp = plot_df[species_name].to_numpy()
                mask = ~np.isnan(c_exp)
                ax.scatter(
                    t_exp[mask], c_exp[mask],
                    color=color, marker='o', s=50, zorder=5, edgecolors='white'
                )

        # 実験データあり時は x 軸を解の時間範囲に合わせ、モデル曲線が確実に視認できるようにする
        if plot_df is not None:
            ax.set_xlim(t_sol_min, t_sol_max)

        # 横軸ラベル: expdata_reader で取得した単位を使う。データがなければ "Time ()"
        xlabel = f"Time ({time_unit})" if time_unit is not None else "Time ()"
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Concentration', fontsize=12)
        plt.title('Chemical Reaction Kinetics - Sample Data', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 描画した化学種についてのみ、最終時刻での濃度を表示
        print("\n=== Concentration at the final time point ===")
        for name, i in plot_items:
            print(f"{name}: {sol.y[i][-1]:.6f}")

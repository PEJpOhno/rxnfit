# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 09/07/2025, M. Ohno

"""
Calculate and visualize the time evolution of chemical species
based on reaction rate equations with all rate constants known.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import sys
import warnings
from sympy import Basic as SympyBasic

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from .build_ode import RxnODEbuild, create_system_rhs


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
    """
    y0: list
    t_span: tuple
    t_eval: Optional[np.ndarray] = field(default=None)
    method: str = "RK45"
    rtol: float = 1e-6


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
        
        Returns:
            tuple: A tuple containing:
                - ode_construct (tuple): ODE system construction data including
                    system_of_equations, sympy_symbol_dict, ode_system,
                    function_names, and rate_consts_dict.
                - solution (scipy.integrate.OdeResult): Solution object from
                    scipy.solve_ivp containing time points and concentrations.
                    May be None if integration fails.
        """
        # 数値積分に必要なオブジェクトを取得
        ode_construct = self.builder.get_ode_system()
        (system_of_equations, sympy_symbol_dict,
         ode_system, function_names, rate_consts_dict) = ode_construct

        # 共通関数を使用して微分方程式の右辺を定義
        system_rhs = create_system_rhs(ode_system, function_names)

        # 数値積分を実行
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

        # 結果を内部に保持
        self.ode_construct = ode_construct
        self.solution = solution
        return ode_construct, solution

    # 結果をプロット
    def solution_plot(self, solution=None, expdata_df=None, species: Optional[List[str]] = None):
        """Plot the time evolution of chemical species.
        
        Creates a time-course plot showing the simulated concentration of
        each species over time. Optionally overlays experimental data
        points with colors matching the simulation lines.
        Also prints the final concentrations at the last time point.

        Args:
            solution (scipy.integrate.OdeResult, optional): Solution
                object from solve_ivp to plot. If None, uses the solution
                stored internally from solve_system(). Defaults to None.
            expdata_df (pandas.DataFrame, optional): Experimental data
                for overlay. DataFrame format:
                - First column: time values.
                - Subsequent columns: concentrations for each species.
                - Column names must match species names (e.g. from
                  self.builder.function_names).
                If provided, scatter points are drawn for each species
                with the same color as the corresponding simulation line.
                Missing values (NaN) are skipped. When provided, the x-axis
                is set to the solution time range so the model curve is
                not compressed when experimental data has a wider range.
                Defaults to None.
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
                    f"次の化学種は微分方程式に現れません: {invalid}. "
                    f"利用可能な化学種: {all_species}. 指定を見直してください.",
                    UserWarning,
                    stacklevel=2,
                )
                raise ValueError(
                    f"微分方程式に現れない化学種が指定されました: {invalid}. "
                    f"利用可能: {all_species}"
                )
            plot_species = list(species)

        # プロットする (名前, 解のインデックス) のリスト
        name_to_idx = {name: i for i, name in enumerate(all_species)}
        plot_items = [(name, name_to_idx[name]) for name in plot_species]

        print("\n=== Time-course plot ===")
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # 解の時間範囲（expdata_df 指定時に軸をここに合わせ、実線が圧縮されないようにする）
        t_sol_min, t_sol_max = float(sol.t.min()), float(sol.t.max())

        for species_name, i in plot_items:
            line, = ax.plot(sol.t, sol.y[i], label=species_name, linewidth=2)
            color = line.get_color()
            if expdata_df is not None and species_name in expdata_df.columns:
                t_exp = expdata_df.iloc[:, 0].to_numpy()
                c_exp = expdata_df[species_name].to_numpy()
                mask = ~np.isnan(c_exp)
                ax.scatter(
                    t_exp[mask], c_exp[mask],
                    color=color, marker='o', s=50, zorder=5, edgecolors='white'
                )

        # expdata_df 指定時は x 軸を解の時間範囲に合わせ、モデル曲線が確実に視認できるようにする
        if expdata_df is not None:
            ax.set_xlim(t_sol_min, t_sol_max)

        plt.xlabel('Time (s)', fontsize=12)
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

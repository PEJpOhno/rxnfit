# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 09/07/2025, M. Ohno

from dataclasses import dataclass, field
from typing import Optional
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from .build_ode import RxnODEbuild


# ODEソルバーへのパラメータ指定  
@dataclass
class SolverConfig:
    y0: list              # 初期濃度（必須）
    t_span: tuple         # 時間範囲（必須）
    t_eval: Optional[np.ndarray] = field(default=None)  # 任意
    method: str = "RK45"  # 任意
    rtol: float = 1e-6    # 任意


class RxnODEsolver:
    def __init__(self, builder: RxnODEbuild, config: SolverConfig):
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
        """反応速度定数の辞書の値が全てfloatかどうかをチェックする
        
        Args:
            rate_consts_dict (dict): 反応速度定数の辞書
            
        Returns:
            bool: 全ての値がfloatの場合はTrue、そうでない場合はFalse
        """
        if not isinstance(rate_consts_dict, dict):
            return False
        
        for key, value in rate_consts_dict.items():
            if not isinstance(value, (int, float)):
                return False
        
        return True


    def solve_system(self):

        # 数値積分に必要なオブジェクトを取得
        ode_construct = self.builder.get_ode_system()
        (system_of_equations, sympy_symbol_dict,
         ode_system, function_names, rate_consts_dict) = ode_construct

        # 微分方程式の右辺を定義
        def system_rhs(t, y):
            """ODEシステムの右辺を計算する関数"""
            rhs_odesys = []
            for i, species_name in enumerate(function_names):
                if species_name in ode_system:
                    try:
                        rhs_odesys.append(ode_system[species_name](t, *y))
                    except Exception as e:
                        print(f"Error in {species_name}: {e}")
                        rhs_odesys.append(0.0)
                else:
                    rhs_odesys.append(0.0)
            return rhs_odesys

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
    def solution_plot(self, solution=None):
        sol = solution if solution is not None else self.solution
        if sol is None:
            print("No solution to plot. Run solve_system() first or pass a solution.")
            return

        unique_species = self.builder.function_names
        print("\n=== Time-course plot ===")
        plt.figure(figsize=(12, 8))

        for i, species_name in enumerate(unique_species):
            plt.plot(sol.t, sol.y[i], label=species_name, linewidth=2)

        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Concentration', fontsize=12)
        plt.title('Chemical Reaction Kinetics - Sample Data', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 最終時刻での濃度を表示
        print("\n=== Concentration at the final time point ===")
        final_concentrations = {name: conc[-1] for name, conc in zip(unique_species, sol.y)}
        for name, conc in final_concentrations.items():
            print(f"{name}: {conc:.6f}")

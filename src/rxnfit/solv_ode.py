# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 09/07/2025, M. Ohno

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.rxnfit.build_ode import RxnODEbuild


# 作成した微分方程式に関する情報を表示
def get_ode_info(rxn_ivp, debug_info=False):    
    print(f"number of species: {len(rxn_ivp.function_names)}")
    print(f"unique species: {rxn_ivp.function_names}")
    print(f"rate constant: {rxn_ivp.rate_consts_dict}")

    if debug_info is True:
        # デバッグ情報を確認
        print("\n=== debug info ===")
        debug_info = rxn_ivp.debug_ode_system()
        print(f"order of args: {debug_info['lambdify_args']}")
        print(f"system of ODE: {debug_info['ode_expressions']}")


@dataclass
class SolverConfig:
    rxn_ivp: RxnODEbuild
    y0: list              # 初期濃度（必須）
    t_span: tuple         # 時間範囲（必須）
    t_eval: Optional[np.ndarray] = field(default=None)  # 任意
    method: str = "RK45"  # 任意
    rtol: float = 1e-6    # 任意


def solve_system(config: SolverConfig):

    # 数値積分に必要なオブジェクトを取得
    #print("\n=== ODEシステムの取得 ===")
    ode_construct = rxn_ivp_build.get_ode_system()
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
    #print("\n=== 数値積分の実行 ===")
    solution = None
    try:
        solution = solve_ivp(
            system_rhs, 
            config.t_span, 
            config.y0, 
            t_eval=config.t_eval,
            method='RK45'
        )
        #print("数値積分が成功しました！")
    except Exception as e:
        print(f"数値積分でエラーが発生しました: {e}")
        print("デバッグ情報を確認してください。")
    return ode_construct, solution



# 結果をプロット
def solution_plot(function_names, solution):
    print("\n=== 結果のプロット ===")
    plt.figure(figsize=(12, 8))
    
    for i, species_name in enumerate(function_names):
        plt.plot(solution.t, solution.y[i], label=species_name, linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.title('Chemical Reaction Kinetics - Sample Data', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 最終時刻での濃度を表示
    print("\n=== 最終時刻での濃度 ===")
    final_concentrations = {name: conc[-1] for name, conc in zip(function_names, solution.y)}
    for name, conc in final_concentrations.items():
        print(f"{name}: {conc:.6f}")
        









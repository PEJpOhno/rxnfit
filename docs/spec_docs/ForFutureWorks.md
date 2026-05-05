## Future worls: 

## 初期濃度を変化させる事例  
https://doi.org/10.1002/ange.201609757  
https://doi.org/10.1039/C8SC04698K  
https://doi.org/10.1039/D4DD00111G  

## 重合反応  
https://doi.org/10.1295/koron.61.394  
https://doi.org/10.1021/ja01267a030  
https://doi.org/10.1039/D6PY00103C  

Shi, J.; He, J.; Yao, Q.; Li, R.; Liu, D.; Liang, X.; Wang, L. Polymerization Reaction Kinetics of Poly α-Olefin and Numerical Simulation of a Continuous Polymerization Reactor. Processes 2025, 13, 3375. https://doi.org/10.3390/pr13113375

**References**  
6. Paul J. Flory, Kinetics of Polyesterification: A Study of the Effects of Molecular Weight and Viscosity on Reaction Rate, Journal of the American Chemical Society 1939 61 (12), 3334-3340, DOI: 10.1021/ja01267a030  
https://doi.org/10.1021/ja01267a030


- **torchdiffeq**: Chen, R. T. Q. (2021). torchdiffeq (Version 0.2.2) [Computer software]. https://github.com/rtqichen/torchdiffeq  

## torchdiffeq  
## rxnfit torchdiffeq 対応 決定仕様 (numbalsoda導入前)

## 1. 全体構成

| 項目 | 決定内容 |
|------|----------|
| 追加モジュール | build_ode_torch.py, solv_ode_torch.py, expdata_fit_torchdiffeq.py |
| ファイル作成 | torchdiffeq 用モジュールは、既存の .py を変更せず、新規に .py ファイルを作成する |
| 既存モジュール | 変更しない |
| 共通利用 | expdata_reader, rate_const_ft_eval, RxnODEbuild（入力として） |
| モジュール関係 | フィット中は expdata_fit_sci が solve_ivp を直接呼ぶのと同様に、expdata_fit_torchdiffeq は RxnODEsolverTorch を使わず torchdiffeq.odeint_adjoint を直接呼ぶ。RxnODEsolverTorch はスタンドアロン積分用 |

## 2. build_ode_torch

| 項目 | 決定内容 |
|------|----------|
| import 元 | build_ode から RxnODEbuild を import する（方法 A） |
| 入力 | RxnODEbuild を引数として受け取る |
| 主な関数 | create_ode_system_with_rate_consts_torch(builder), create_system_rhs_torch(...) |
| create_ode_system（resolve 版） | 実装しない。経時曲線作成時は、フィットした k を rate_const_values に dict で渡す |
| SymPy → PyTorch | lambdify の modules に torch 用マッピング辞書（カスタム modules） |
| 例外処理 | lambdify 失敗時は RuntimeError を送出（fail fast） |
| 対応 SymPy 関数 | 加減乗除、指数対数（exp, log）、シグモイド。sqrt は x**0.5（Pow）で表現 |

## 3. solv_ode_torch

| 項目 | 決定内容 |
|------|----------|
| 役割 | torchdiffeq による ODE 積分（スタンドアロン用。フィッティング中は expdata_fit_torchdiffeq が odeint_adjoint を直接呼ぶため本モジュールは使わない） |
| 主なクラス | RxnODEsolverTorch, SolverConfigTorch |
| SolverConfig | 別クラス（SolverConfigTorch）。solv_ode_torch.py に定義（solv_ode.py と同様の構成） |
| 積分器 | torchdiffeq.odeint_adjoint のみ（odeint は使わない。メモリ効率・訓練向け）。RHS を nn.Module で包むか adjoint_params でパラメータを渡す必要がある |
| 積分手法（method） | 入力は scipy 名（RK45 等）。内部で torchdiffeq 名（dopri5 等）に変換 |
| method マッピング | 下表（scipy 公式・torchdiffeq README に基づく） |
| デフォルト method | dopri5（torchdiffeq のデフォルトに合わせる） |

#### method マッピング（scipy 名→torchdiffeq 名）

| scipy (solve_ivp) | torchdiffeq | 根拠 |
|-------------------|-------------|------|
| RK45 | dopri5 | いずれも Dormand-Prince 5(4) |
| RK23 | bosh3 | いずれも Bogacki-Shampine 3(2) |
| DOP853 | dopri8 | いずれも Dormand-Prince 8次 |
| Radau | radauIIA5 | いずれも Radau IIA 5次 |
| BDF | scipy_solver | torchdiffeq に BDF 相当なし。scipy_solver でラップ |
| LSODA | scipy_solver | scipy_solver でラップ |

| 項目 | 決定内容 |
|------|----------|
| 解の形式 | (t, y) の tuple |
| t_eval | t_eval を指定しなかった場合は np.linspace(t0, t1, 100) をデフォルトグリッドとする。t_eval は引数として設定可能（SolverConfigTorch 等）。実装ではグリッドを torch.Tensor に変換して odeint_adjoint に渡す |
| 付随メソッド | solution_plot, rsq, to_dataframe は提供しない。経時曲線作成時はフィットした k を rxnfit の RxnODEsolver に渡して解く。付随メソッドは rxnfit のものをそのまま使う |

## 4. expdata_fit_torchdiffeq

| 項目 | 決定内容 |
|------|----------|
| 役割 | ExpDataFitSci 相当のフィッティング |
| 主なクラス/関数 | ExpDataFitTorchdiffeq, run_fit_multi_torchdiffeq |
| 最適化 | torch.optim（Adam）のみ。scipy による最適化が必要な場合は既存の expdata_fit_sci を使用する |
| オプティマイザのデフォルト | Adam（torch.optim。scipy 版の opt_method とは別） |
| lr（学習率） | torch.optim.Adam 等に引数として渡す仕組みを設ける。デフォルトは PyTorch のデフォルト（1e-3）に合わせる（torchdiffeq は lr を持たないため、torch.optim と同様とする） |
| 積分器 | フィット中は RxnODEsolverTorch を使わず torchdiffeq.odeint_adjoint を直接呼ぶ（expdata_fit_sci が solve_ivp を直接呼ぶのと同様）。odeint_adjoint のみ。RHS を nn.Module で包むか adjoint_params でフィット対象パラメータを渡す |
| 結果の形式 | rxnfit と統一（OptimizeResult 相当） |
| param_info | expdata_fit_sci と同じ構造 |
| メソッド | get_solver_config_args, get_fitted_rate_const_dict 相当を提供 |
| get_solver_config_args | 戻り値は rxnfit の SolverConfig 用の kwargs（y0, t_span, method, rtol 等）。経時曲線作成で RxnODEsolver に渡す |
| get_fitted_rate_const_dict | 戻り値は dict（{symbolic_key: value}）。expdata_fit_sci と同じ形式 |

## 5. デバイス（CPU/GPU）

| 項目 | 決定内容 |
|------|----------|
| 方針 | torchdiffeq に合わせる |
| 扱い | y0, t, ODE 関数の入出力が置かれたデバイスで計算。呼び出し側が指定 |
| 設定 | SolverConfigTorch 等に device パラメータを持たせ、y0, t をそのデバイスに統一 |

## 6. 例外処理

| 項目 | 決定内容 |
|------|----------|
| build_ode / build_ode_torch | lambdify 失敗時は RuntimeError を送出（fail fast） |
| create_system_rhs | func が None または評価失敗時は RuntimeError を送出 |
| rxnfit との統一 | rxnfit を fail fast に変更済み |

## 7. dtype

| 項目 | 決定内容 |
|------|----------|
| デフォルト | float32（GPU を優先） |

## 8. bounds / use_log_fit

| 項目 | 決定内容 |
|------|----------|
| bounds | 下限制約のみ。パラメータ変換は softplus（k = low + softplus(θ)） |
| use_log_fit | scipy 版と同じ API。log 空間で最適化し、ODE では exp で変換 |
| use_log_fit と bounds | scipy 版に合わせる（use_log_fit 時は bounds を無視） |
| lower_bound デフォルト | 1e-14。float32 時は内部で 1e-7 をフロアとするか、ドキュメントで注意 |

## 9. 依存関係

| 項目 | 決定内容 |
|------|----------|
| torchdiffeq / PyTorch | 任意（optional）の dependencies とする |
| インストール | extras_require で選択式。例: `pip install rxnfit[torch]` |
| import 失敗時 | torch が無い状態で torch 用モジュールを import した場合、ImportError でよい |
| __init__.py | torch 用モジュールは __init__.py から import しない。`from rxnfit.build_ode_torch import ...` 等で直接指定して使用 |

## 10. 前提・rxnfit 側

| 項目 | 決定内容 |
|------|----------|
| 半反応（生成物のみ） | rxnfit の改修で対応済み。torchdiffeq モジュールでは追加対応不要 |
| 時間依存 k(t) | rxnfit の既存機能を利用。torch 版では evaluator の出力を tensor に変換して RHS に渡す |

---

## 実装における不明瞭な点（要検討）

### expdata_fit_torchdiffeq
| 項目 | 不明点 |
|------|--------|
| 収束判定 | 以下は指定がない場合のデフォルト値。引数で変更可能にする。最大 iter 数、loss 閾値は保留。early stopping は有 |

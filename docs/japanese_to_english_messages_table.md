# .py ファイル内 日本語のエラーメッセージ・ログ・出力 → 英語修正箇所一覧

**対象**: エラーメッセージ（raise）、ログ・出力（print, warnings.warn）に含まれる日本語  
**対象外**: コメント（`#` 以降）は日本語のまま残す

---

| ファイル | 行 | 種別 | 現在の文言（日本語） | 修正案（英語） |
|----------|-----|------|----------------------|----------------|
| **expdata_fit_sci.py** | 65-67 | ValueError | パラメータが不足しています。速度定数 {key} の値が必要です。 | Insufficient parameters. A value for rate constant {key} is required. |
| expdata_fit_sci.py | 79 | ValueError | tは配列である必要があります | t must be an array. |
| expdata_fit_sci.py | 91-93 | RuntimeError | 数値積分が失敗しました: {solution.message} | Numerical integration failed: {solution.message} |
| expdata_fit_sci.py | 96 | print | 数値積分中にエラーが発生しました: {e} | An error occurred during numerical integration: {e} |
| expdata_fit_sci.py | 134-136 | ValueError | パラメータが不足しています。速度定数 {key} の値が必要です。 | （上と同じ）Insufficient parameters. A value for rate constant {key} is required. |
| expdata_fit_sci.py | 226-228 | ValueError | fixed_initial_valuesの長さ({len(fixed_initial_values)})が化学種数({len(function_names)})と一致しません。 | len(fixed_initial_values) does not match the number of species. |
| **p0_opt_fit.py** | 59-61 | ValueError | param_bounds にシンボリック速度定数に存在しないキーが含まれています: {key}. 使用可能: {sorted(allowed)}. | param_bounds contains a key that is not a symbolic rate constant: {key}. Valid keys: {sorted(allowed)}. |
| p0_opt_fit.py | 67-68 | TypeError | param_bounds は (下限, 上限) のタプル、または変数名をキーとする辞書である必要があります。 | param_bounds must be a (low, high) tuple or a dict with variable names as keys. |
| p0_opt_fit.py | 134-136 | ValueError | シンボリックな速度定数がありません。フィッティング対象が存在しません。 | No symbolic rate constants. Nothing to fit. |
| p0_opt_fit.py | 181-183 | RuntimeError | run_fit が収束しませんでした: {message} | run_fit did not converge: {message} |
| p0_opt_fit.py | 231 | RuntimeError | 全トライアルが失敗しました。 | All trials failed. |
| **build_ode.py** | 255-257 | RuntimeError | ODE構築に失敗しました (化学種: {key}). 式: {expr}. 原因: {e} | ODE construction failed (species: {key}). Expression: {expr}. Cause: {e} |
| build_ode.py | 331-333 | RuntimeError | （上と同じ）ODE構築に失敗しました (化学種: {key}). 式: ... 原因: {e} | （上と同じ）ODE construction failed (species: {key}). Expression: ... Cause: {e} |
| build_ode.py | 474-476 | RuntimeError | 化学種 '{species_name}' のODE関数がNoneです。lambdifyの失敗が想定されます。 | ODE function for species '{species_name}' is None. lambdify likely failed. |
| build_ode.py | 514-516 | RuntimeError | ODE評価中にエラーが発生しました (化学種: {species_name}). 原因: {e} | An error occurred during ODE evaluation (species: {species_name}). Cause: {e} |
| **rxn_reader.py** | 368-370 | ValueError | 式の右辺に存在しない速度定数を指定しました: '{parsed.name}'. 使用可能な速度定数: {sorted(allowed_keys)} | Rate constant '{parsed.name}' in expression is not defined. Allowed rate constants: {sorted(allowed_keys)} |
| rxn_reader.py | 374-376 | ValueError | 式の右辺に存在しない速度定数を指定しました: {undefined}. 使用可能な速度定数: {sorted(allowed_keys)} | Rate constants {undefined} in expression are not defined. Allowed: {sorted(allowed_keys)} |
| **expdata_reader.py** | 111-113 | warnings.warn | 経時変化データの0列目（時間列）の列名が一致しません: {names_0}. 先頭のDataFrameの列名を用います。 | Time column (0th) names differ across DataFrames: {names_0}. Using the first DataFrame's column name. |
| expdata_reader.py | 178-179 | ValueError | 化学種 '{name}' がデータの列に見つかりません。 | Species '{name}' not found in DataFrame columns. |
| expdata_reader.py | 212 | ValueError | 化学種 '{name}' がデータの列にありません。 | Species '{name}' is not in the data columns. |

---

## 補足

- **expdata_fit_sci.py**: run_fit 内の p0 関連の ValueError および verbose 時の print は、すでに英語に変更済みのため表には含めていません。
- **build_ode.py**: 33行目以降の ValueError / TypeError（rate constant overrides 系）はもともと英語のため未掲載です。
- **solv_ode.py**: raise / print / warnings は英語のため未掲載です。
- **rate_const_ft_eval.py**: 該当する日本語のエラー・ログ・出力はありません。

# 現行の機能はそのままで、k を t の関数とする機能を追加する

反応式定義の CSV の速度定数列で文字列として記号を与え、別の CSV でその記号列および式の定義の列に分ける。  
この列名は `k`, `f(t)` とする。

ただし、現行の他の速度定数で他の速度定数を定義する場合の式（例: `k2 = k1/4`）をこの CSV に書き込んだ場合も、現状と同様に動作するように実装する。

---

# 速度定数定義 CSV の仕様

- **1 行目（ヘッダー）**  
  `"k"`, `"f(t)"` を必須とし、厳密にこの 2 列のみ受け付ける。

- **コメント行**  
  `#` で始まる行は認めず、エラーとする。

- **k の重複**  
  同じ `k` が複数行ある場合はエラー。

- **f(t) が空**  
  その行は無視し、反応 CSV の定義のままフィッティング対象とする。

- **空行**  
  無視する。

- **反応 CSV に存在しない k**  
  例: `k99` のように反応 CSV にないキーが書かれていた場合はエラー。

---

# その他の仕様

## 速度式定義ファイルの扱い

- 速度式定義の別ファイルの有無は  
  **「案1: 引数は常に省略可能にして、渡したかどうかで区別する」** とする。
- 引数名は `rate_const_overrides` で統一し、`RxnODEbuild` のコンストラクタに追加する。
- `rate_const_overrides_encoding` は、`rate_const_overrides` がファイルパス（str）のときだけ有効と docstring に記載する。
- 引数として path を渡す場合は **CSV のみサポート**し、それ以外はエラー。

---

## 自由パラメータの扱い

- 自由パラメータ（`symbolic_rate_const_keys` やフィットする変数）を集めるとき、  
  **シンボル t は除外**する。  
  t は積分の独立変数としてのみ扱う。

- 別 CSV の式をパースしたとき、既知のもの（反応で定義された k のシンボルと t）以外のシンボルは、  
  **フィット対象の自由パラメータに含める**。

---

## overrides の適用タイミング

- 反応 CSV 読み込み直後に overrides を適用し、  
  `rate_consts_dict` および `sympy_symbol_dict` の両方を更新する。

---

## k(t) を含むモデルでの SolverConfig 引数生成

- `rate_const_values` / `symbolic_rate_const_keys` を組み立てる処理は  
  **run_fit() 実行後に行う**。

- run_fit() 前に呼んだ場合はエラーとし、メッセージは以下とする：



---

# プロット関連の仕様

- `SolverConfig` に `rate_const_values` と `symbolic_rate_const_keys` の 2 項目を追加する。

- `RxnODEsolver.solve_system` では、これらが指定されているときだけ  
  **速度定数つき ODE** として  
  `create_system_rhs(..., rate_const_values=..., symbolic_rate_const_keys=...)` を使う。

- `solve_ivp` には  
  `fun, y0, t_span, t_eval, method, rtol` のみを渡す。

- `create_system_rhs` で `rate_const_values` が callable のときは  
  `rate_const_values(t)` でその時点の dict を取得する。

- k が定数か k(t) かは自動判別し、その結果に応じて  
  `rate_const_values` と `symbolic_rate_const_keys` を SolverConfig に渡す。

- `rate_const_values` と `symbolic_rate_const_keys` は  
  **両方指定するか両方省略するか** のどちらかとし、片方だけ指定の場合はエラー。

- `rsq()` など積分を行うメソッドも、config に rate が指定されているときは  
  **速度定数つき ODE 経路** を使う。

---

# 評価器モジュール

- 評価器は `rate_const_ft_eval.py` として別モジュールに置く。  
- `ft` が `f(t)` の略であることは docstring に記載する。

---

# 反応 CSV の速度定数列の扱い

- 反応 CSV の速度定数列に `k2` などと書いた場合、  
  その k の定義が別 CSV に存在すればそこで上書きする。

- 存在しなければ現行どおり `Symbol("k2")` のまま（未定義のシンボル）。

- 反応 CSV の速度定数が空欄の場合は、現行通りまず強制的に `k+RID` を設定する。

---

# (B) 修正・追加モジュール一覧

| 種別 | モジュール / ファイル | 変更・追加内容 |
|------|------------------------|----------------|
| 修正 | rxnfit/build_ode.py | - `RxnODEbuild.__init__` に `rate_const_overrides=None`, `rate_const_overrides_encoding=None` を追加。<br>- override CSV の読み込みを build_ode 内に実装（厳密ヘッダー、空行無視、# エラー、k 重複エラー、未定義 k エラー）。<br>- 読み込み直後に overrides を適用し dict を更新。<br>- 自由パラメータ構築時に t を除外。<br>- `create_system_rhs()` で callable の場合 `rate_const_values(t)` を使用。 |
| 修正 | rxnfit/solv_ode.py | - SolverConfig に `rate_const_values`, `symbolic_rate_const_keys` を追加。<br>- 両方指定 or 両方省略のみ許可。<br>- solve_system で速度定数つき ODE 経路を追加。<br>- rsq など積分系も同様。 |
| 修正 | rxnfit/expdata_fit_sci.py | - `k(t)` の有無を自動判別。<br>- run_fit() 後のみ SolverConfig 引数を返す。<br>- 未実行時は指定メッセージでエラー。<br>- 評価器は rate_const_ft_eval を使用。 |
| 追加 | rxnfit/rate_const_ft_eval.py | - 新規モジュール。<br>- docstring に「ft = f(t)」。<br>- `(t, params)` → rate dict を返す callable を構築。 |
| 追加 | （任意）rxnfit/__init__.py | - 必要に応じて公開 API を export。 |

---

# Overview

"rxnfit" is a Python module to set up theoretical reaction kinetics equations from reaction formulas and the rate constants provided as csv file.

- 速度定数がすべて既知の場合、構築した反応速度式から各化学種の濃度の経時変化を出力できる。
- 未知の速度定数があり、化学種の経時変化データがある場合、フィッティングにより未知の速度定数を推定できる。
- ある素反応の速度定数を別の素反応の速度定数の関数、または時間 t の関数として表現できる。
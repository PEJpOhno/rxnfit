# numba / numbalsoda 高速化仕様書

## 1. 概要

- `rxnfit` の ODE 計算（およびフィット由来の多数回積分）を `numba` / `numbalsoda` で高速化する。
- 既存 API と互換性を優先し、破壊的変更を避ける。
- 実行時に SciPy と numbalsoda を常時比較しない。数値同等性は CI で検証する。
- `numbalsoda` は optional。利用可能なときだけ高速化し、不可のときは `scipy` にフォールバックする。
- 責務の集約: `solver_backend`（ディスパッチ・警告・`t_eval` 補完ヘルパー）、`numbalsoda_rhs.py`（`cfunc` 等の重い RHS 生成）、`lsoda_p_vector.py`（`p` ベクトル組み立て・公開モジュール）。

### 1.1 上流と接続

- 反応系の数学表現の単一ソース: **SymPy → `lambdify`（numpy）→ `create_system_rhs`**。numbalsoda 用 RHS はここから機械的に導出する（手書き二重管理はしない）。
- `solver_backend.solve_ode`: `method="LSODA"` のときのみ numbalsoda を試行し、失敗時は初回警告のうえ `RK45` にフォールバック。lazy import と import 結果のプロセス内キャッシュを用いる。
- パッケージング: `pyproject.toml` の **`[project.optional-dependencies]` の `nlsoda` extra** に `numbalsoda` を置く。必須依存には含めない。
- SciPy は **`dy/dt` を戻り値**で受け取る。numbalsoda（LSODA）は **`rhs(t, u, du, p)` で `du` に書き込む** ABI。生成器がこの差を吸収する。
- `create_system_rhs` が返す **`system_rhs(t, y)` は通常の Python callable で `.address` を持たない**。numbalsoda 経路は **`numba.cfunc` 由来のネイティブ `.address`** を要するため、生成レイヤで **`system_rhs.numbalsoda_rhs`** に接続する（未対応時の扱いは §4）。

---

## 2. 依存関係

- `numbalsoda` は optional。`pip` と `conda` で requirement が異なる前提を受け入れ、単一の統一定義は行わない。
- 導入手順はパッケージマネージャごとにドキュメントで示す。導入可否は利用者が判断する。
- 実行時は import 可否で backend を選択する。`numba` 等の細かいバージョン縛りは numbalsoda とパッケージマネージャの解決に任せる。
- **README にも `pyproject.toml` にも**、numbalsoda／numba のバージョン組み合わせを「動作確認済み例」として列挙・推奨しない（§6 と整合）。

### 2.1 依存を numbalsoda 側に任せるときの得失

**利点**: rxnfit の `pyproject.toml` が Numba/LLVM 系のピン留めで肥大化しにくい。PyPI / conda-forge の選択がしやすい。numbalsoda 側の追随で rxnfit の更新が減ることがある。

**欠点**: 同一 rxnfit でも numbalsoda/numba の組み合わせで JIT 初回時間や数値の微小差が変わり得る。依存衝突の issue がスタック側に見えがち。optional を CI で広く回さないと特定組み合わせのみの欠陥を検出しにくい。

---

## 3. method と時間依存経路

- backend モードは増やさず、`method` で経路を決める。`method="LSODA"` のときだけ numbalsoda を試行する。それ以外は `scipy.integrate.solve_ivp`（指定 method）。`RK45` 等は numbalsoda 内では扱わない。

**時間依存速度定数経路**（numbalsoda を使わない）は次のいずれか。

1. `rate_const_values` が **`callable`**（`t` を受け `dict` を返す）として `create_system_rhs` に渡される場合（例: `expdata_fit` の `evaluator(t, params)`）。
2. `rate_consts_dict` に対し **`has_time_dependent_rates` が真**（いずれかのレート式の `free_symbols` に名前 `'t'` のシンボルが含まれる）。`k`/`f(t)` 別 CSV や反応 CSV 由来を含む。

上記に該当するとき ODE 積分は **`solve_ivp` + `method="RK45"`** に固定する。

**LSODA を numbalsoda で高速化する対象**は、上記 1・2 がいずれも偽で、各積分の間に速度定数が数値固定の経路に限る。

**分岐の置き場**: `solver_backend.solve_ode` の入口で、`method="LSODA"` かつ時間依存に該当するなら numbalsoda を試行せず `RK45` の SciPy 経路へ回す。

### 3.1 `time_dependent` と `solve_ode` のシグネチャ

- `solve_ode` に **`time_dependent: bool = False`** を追加する。利用者が手で設定しない。`solv_ode` / `expdata_fit` 等が **`callable` または `has_time_dependent_rates`** から自動で決めて渡す。`k`/`f(t)` 別 CSV で `has_time_dependent_rates` が真なら、ユーザー操作なしで `time_dependent=True`。
- `solve_ode` 内では `rate_consts_dict` を再構築して SymPy 判定は行わない。

**公開シグネチャ**（`**kwargs` にまとめない）: 位置引数 `system_rhs, t_span, y0`、キーワードはこの順 — `t_eval=None, method="RK45", rtol=1e-6, atol=None, time_dependent=False`。

### 3.2 `method="LSODA"` と時間依存の併用時の警告

- `method="LSODA"` かつ `time_dependent=True` のとき、numbalsoda は試行しない。`solve_ivp` + `RK45` に切り替えて継続する。
- 切り替え時は初回のみ（プロセス単位）`warnings.warn(..., UserWarning)`。キーは §4 のフォールバックと別（二重通知を避ける）。
- **警告キー**: `lsoda_time_dependent_coerce`（§4 の表）。
- **英文**（改変しない）:  
  `method=LSODA in rxnfit does not support time‑dependent rate constants, the solver should be switched to RK45.`

### 3.3 `time_dependent=True` 強制 RK45 分岐内の `rtol` / `atol` / `t_eval` / `t_span`

- `rtol` は呼び出し元の値を `solve_ivp` にそのまま渡す。
- `atol` が未指定なら SciPy 既定に任せる（rxnfit が LSODA 専用の独自既定 atol を新設しない）。
- `t_eval` / `t_span` は **§5 の LSODA 時間軸ルールを適用しない**（呼び出し元・SciPy の通常の解釈のまま `solve_ivp` に渡す）。

### 3.4 将来境界（時間依存の検出）

- §3 先頭の 1・2 のいずれでも検出できない形の時間依存表現は許容しない。想定外の経路が見つかった場合はバグ修正と README 追記を基本とし、本書では個別手順を固定しない。

---

## 4. フォールバックと警告

`method="LSODA"` かつ `time_dependent=False` のとき、numbalsoda 経路を完走できない場合は **例外にせず `RK45`（`solve_ivp`）にフォールバック**する。初回のみ（プロセス単位）`warnings.warn(..., UserWarning)`。警告キーは経路ごとに分ける（§3.2 の時間依存警告とも文面を分ける）。

### 4.1 警告キー一覧

`_warn_once` 等の初回抑止に用いるキー文字列。実装で改名しない。

| キー | 条件の要約 |
|------|------------|
| `lsoda_unavailable` | numbalsoda の import 失敗・未インストール等（§4.2） |
| `lsoda_rhs_unsupported` | import は成功したが RHS が numbalsoda 未対応（§4.3） |
| `lsoda_run_failed` | RHS 接続後の `lsoda` 実行失敗（§4.4） |
| `lsoda_time_dependent_coerce` | §3.2 |

### 4.2 パッケージ利用不可（`lsoda_unavailable`）

- numbalsoda の import ができない場合。
- 文面: `LSODA is not available in your environment. Falling back to RK45.`

### 4.3 RHS が numbalsoda 未対応（`lsoda_rhs_unsupported`）

- import は成功したが、`system_rhs.numbalsoda_rhs` が無い、または **ネイティブ `.address` が無く** LSODA 経路に載せられない場合。
- **`numbalsoda_rhs` 生成**で Numba JIT / `cfunc` コンパイル等が失敗し、利用可能な `.address` が得られない場合も **同じ扱い**（`lsoda` 呼び出し前の失敗。§4.4 と分離）。
- 文面（改変しない）:  
  `Though the import numbalsoda succeeded, RHS is not supported.`
- 生成が JIT で失敗した場合は **`lsoda` を試行せず**本節のフォールバックとする。

### 4.4 `lsoda` 実行失敗（`lsoda_run_failed`）

- import は成功し、`system_rhs.numbalsoda_rhs` に有効な `.address` があり **`lsoda` を呼び出した後**、失敗ステータス（例: `success=False`）または Python 例外で終わる場合。§4.3 は含めない。
- 例外にせず `RK45` にフォールバック。初回のみ §4.2 と**同一英文**で警告。
- **`success=False` と Python 例外で警告キーを分けない**（いずれも `lsoda_run_failed`・同一英文）。将来分ける場合は仕様・表・README・実装を同時に改訂する。

その他:

- `method` が `LSODA` 以外は従来どおり `solve_ivp` + 指定 method。
- numbalsoda の import 可否は初回判定後にキャッシュする。

---

## 5. LSODA 時の時間軸（`t_eval` と `t_span`）

**適用範囲**: 本節は **`method="LSODA"` かつ `time_dependent=False`** で LSODA／numbalsoda 経路を解く積分に限る。`t_eval` が与えられているとき **`t_span` は積分区間の決定に用いない**（§5.2）はこの文脈のみ。

**適用しない経路**: `time_dependent=True` で最初から `RK45`（`solve_ivp`）に振られる分岐では本節を適用しない。`t_span` / `t_eval` は SciPy の通常の意味（§3.3）。

**フォールバック後**: §4 により `LSODA` 試行から `RK45` に落ちた場合も、元は時間非依存の LSODA 経路であるため **§5.2**（実効 `t_span` を `t_eval` の端に合わせる等）が引き続き適用される。

**格子の作り方**: `method="LSODA"` かつ時間非依存でも、`t_eval` の作り方は SciPy 経路と同一の **§5.3**（エントリ別の式）に従う。バックエンドを numbalsoda に替えても密格子・`np.unique` の方針は変えない。numbalsoda は **明示的に与えられた `t_eval` 格子**上で解く。

### 5.1 基本原則

- `t_eval` が非 `None` の 1 次元配列で渡された場合はそれを正とし上書きしない。§5.3 に書くエントリ別の自動補完は **`t_eval is None` のときのみ**。
- `method="LSODA"` かつ `t_eval` 未指定のときだけ、エントリごとに §5.3 の規則で補完する（誰がいつ呼ぶかは §5.5）。
- `t_span` を無条件に無視する仕様にはしない（未指定補完の前段などで区間の根拠に使う場合がある）。

### 5.2 `t_eval` が与えられているときの実効 `t_span`

- この場合ユーザーが渡した `t_span` は積分区間の決定に用いない。
- 実効区間は **`t_eval` を昇順にした min / max**。
- `numbalsoda.lsoda` および RK45 フォールバック時の `solve_ivp` に渡す `t_span` はこの実効区間に揃える（呼び出し元の `t_span` と端点が食い違ってよい）。
- 利用者向け: LSODA では時間軸は **`t_eval` で指定**し、`t_eval` ありでは `t_span` は参照されない。

### 5.3 `t_eval` が未指定のときの自動補完（エントリ別）

各エントリは現行実装と同じ時間点の作り方に合わせる。numbalsoda 経路は **`t_eval` が常に非 `None` の配列**を要するため、`method="LSODA"` かつ `t_eval is None` のときだけ §5.5 の共有ヘルパーで格子を生成してから `solve_ode`／numbalsoda に渡す。

1. **`RxnODEsolver.solve_system`**（実験オーバーレイなし）  
   `n_dense = max(100, 1)`、`t_eval = np.linspace(t_span[0], t_span[1], n_dense)`。ここでは **`t_span` が補完区間の根拠**。

2. **`RxnODEsolver._compute_rss`**  
   `t_eval = np.sort(np.unique(t_exp_col))`（密格子は付与しない）。構築後は §5.2。

3. **`expdata_fit._integrate_datasets_for_params`**  
   `n_dense = max(100, len(t_exp) * 10)`、`t_dense = np.linspace(t_start, t_span[1], n_dense)`、`t_eval = np.sort(np.unique(np.concatenate([t_exp, t_dense])))`。構築後は §5.2。

4. **`expdata_fit._eval_ode_fit`**  
   `t_eval = np.array(t_points)`。構築後は §5.2。

### 5.4 他 method

`method != "LSODA"` のときは `t_span` / `t_eval` は SciPy の仕様どおり。`t_eval is None` のまま `solve_ivp` に渡す。§5.5 の共有ヘルパーは呼ばない。

### 5.5 `t_eval` 補完と `solve_ode` の分担

- `solve_ode` は `t_eval is None` のまま §5.3 のコンテキスト分岐を引数だけで再現しない。
- `t_eval is None` の配列化は **`solver_backend` の共有ヘルパーにのみ**実装する。呼ぶのは **`method="LSODA"` かつ `t_eval is None`** のときに限る（numbalsoda 試行前）。`method != "LSODA"` ではヘルパーを挟まない。
- `solv_ode` / `expdata_fit` 等は上記条件のとき **`solve_ode` 直前に必ずヘルパーを呼び**、返した 1 次元 `t_eval` を `solve_ode` に渡す。既に非 `None` ならヘルパーをスキップ可。各エントリに §5.3 の式を重複実装しない。
- `solve_ode` の LSODA 分岐内で §5.2 の実効 `t_span` とフォールバック時の `solve_ivp` 引数整合を行う。

#### 5.5.1 共有ヘルパー: モードと必須入力

1 回の呼び出しでモードは 1 つ。モード識別子は下表の 4 つに固定する（`Enum` メンバ名または同名文字列。実装で改名しない）。生成式は §5.3 の対応行と同一。

| モード | 主な呼び出し元 | 必須入力（補完用） | 生成式 |
|--------|----------------|-------------------|--------|
| `SOLVE_SYSTEM` | `RxnODEsolver.solve_system` | `t_span`: `tuple[float, float]` | `n_dense = max(100, 1)`、`t_eval = np.linspace(t_span[0], t_span[1], n_dense)` |
| `COMPUTE_RSS` | `RxnODEsolver._compute_rss` | `t_exp_col`: 実験時刻 1 次元配列 | `t_eval = np.sort(np.unique(t_exp_col))` |
| `INTEGRATE_DATASETS` | `expdata_fit._integrate_datasets_for_params` | `t_span`、`t_start`、`t_exp` | `n_dense = max(100, len(t_exp) * 10)`、`t_dense = np.linspace(t_start, t_span[1], n_dense)`、`t_eval = np.sort(np.unique(np.concatenate([t_exp, t_dense])))` |
| `EVAL_ODE_FIT` | `expdata_fit._eval_ode_fit` | `t_points` | `t_eval = np.array(t_points)` |

- `EVAL_ODE_FIT`: 多くの経路では先に `np.array(t)` 済みで `solve_ode` に `None` を渡さない。`None` 補完が必要なときに本モードを使う。
- 戻り値: 浮動小数 `dtype` の 1 次元 `ndarray`、昇順かつ重複なし。入力に重複・非昇順があればヘルパー内で `np.sort(np.unique(...))` してよい。
- 不足入力・不正 `t_span` 等: `ValueError`。
- 5 つ目のモードは現仕様では定義しない。ヘルパーはモード列挙を拡張しやすい実装（`Enum`、分岐の集約、`match` 等）にする。

#### 5.5.2 公開 API 名

- 関数名: **`build_t_eval_when_none_for_lsoda`**（`solver_backend` の公開関数。実装で改名しない）。
- モード列挙型: **`LsodaImplicitTEvalMode`**（メンバ名は上表と同名: `SOLVE_SYSTEM`, `COMPUTE_RSS`, `INTEGRATE_DATASETS`, `EVAL_ODE_FIT`）。
- 引数: `**kwargs` にまとめない。第 1 引数（位置）`mode: LsodaImplicitTEvalMode`、キーワードのみ `t_span=None, t_start=None, t_exp_col=None, t_exp=None, t_points=None` の順。モードごとの必須組は表に従い、未使用は `None`、不足・矛盾は `ValueError`。
- `solve_ode` のシグネチャは §3.1。

### 5.6 `t_eval` が空配列のとき

- `len(t_eval)==0` かつ `method="LSODA"` のときは numbalsoda を試行せず `solve_ivp`（例: `RK45`）へ。§4 の警告キーは発火しない通常分岐でよい（専用警告は要求しない）。
- `t_eval is None`（未指定）と長さ 0 の配列は別物。前者は §5.3 の補完があり得る。

### 5.7 呼び出し元の `t_eval` 正規化

- `solve_ode` に渡す前に、呼び出し元で昇順かつ重複なしに正規化する（例: `np.sort(np.unique(...))`）。黙示的に `solve_ode` 入口だけに寄せる設計は契約としない（入口で防御的に整えることは実装の自由）。

---

## 6. モジュール構成・RHS・`p`

### 6.1 `solver_backend`

- import 判定、利用可否、backend 選択、実行とフォールバック、初回警告。
- `method="LSODA"` 時の `t_eval` / 実効 `t_span`（§5）および §5.5 の共有ヘルパー。

### 6.2 `p` 組み立て（公開）

- **`src/rxnfit/lsoda_p_vector.py`** に **`build_lsoda_parameter_vector_p`** を置く（公開関数・実装で改名しない）。`from rxnfit.lsoda_p_vector import build_lsoda_parameter_vector_p` で直接 import 可。
- `solver_backend` の LSODA 分岐は **`numbalsoda.lsoda` 直前**に `build_lsoda_parameter_vector_p` を呼ぶ。

### 6.3 RHS の分割（ハイブリッド）

- 公開インターフェースは **`build_ode.create_system_rhs`**（呼び出し側 API は変えない）。
- **`numbalsoda_rhs.py`**: `cfunc` 生成、シグネチャ、numbalsoda 向け接続。`create_system_rhs` は SciPy 用 callable を組み立てたうえで、必要時のみ `numbalsoda_rhs` を遅延 import して付与する。
- **ルート** `system_rhs(t, y)` は SciPy 向け callable のまま。**`numba.cfunc`（`.address` 保有）は `system_rhs.numbalsoda_rhs` にのみ**付与。ルートを `cfunc` で置き換えたり、ルートに直接 `.address` を付けない。
- §4.3 の未対応: `getattr(system_rhs, "numbalsoda_rhs", None)` が欠けるか、その `.address` が欠けるとき。
- **循環 import**: `numbalsoda_rhs` → `build_ode` の依存は一方向（通常は `build_ode` → `numbalsoda_rhs` のみ）。
- **SymPy／`lambdify` からの自動生成のみ**。手書き Numba カーネルは採用しない。
- §3 の時間依存経路では numbalsoda 用 RHS の生成・利用を行わない。

### 6.4 パラメータベクトル `p` の並び

**目的**: `cfunc` が `p[i]` を読む順と、積分前に `p` を埋める側の順を一致させる。`build_lsoda_parameter_vector_p` と `numbalsoda_rhs` 生成物の docstring に**同じ段落**（ルール・記号・例）を載せる。

- `n_fitted = len(symbolic_rate_const_keys)`。
- `p[0] … p[n_fitted-1]`: `symbolic_rate_const_keys` の順そのまま。`create_system_rhs`／フィッターへのキー順と一致。
- `p[n_fitted] … p[n_p-1]`: ビルダーの速度定数集合から `symbolic_rate_const_keys` を除いた残りについて、キー文字列を **`sorted`（辞書順）** で並べた順。**v1 では自然順（natural sort）に変更しない**。将来変更する場合は実装と docstring を同時更新する別仕様。
- `n_p`: 上記で定まる長さ。同一ビルド／同一 RHS 生成物で不変。

**docstring 三要素**: (1) 上記ルールの要約、(2) `n_fitted`・`n_p` の定義、(3) 例 `symbolic_rate_const_keys = ("k2", "k1")`、残り `k3,k4` のみ → `p = [v_k2, v_k1, v_k3, v_k4]`。

**`u` との関係**: 濃度ベクトル `u` の成分順（`function_names` 順）と `p` の並びを一致させる必要はない。

### 6.5 初版の RHS 生成スコープ

- 固定レート・固定種数に限定。
- numbalsoda／`cfunc` 生成の対象は **`create_system_rhs` の標準形・時間非依存 RHS**（§3 先頭の 1・2 がいずれも偽、積分間は速度定数が数値固定）に限る。
- **対象外の例**（numbalsoda を試行せず §3 の SciPy 経路）: `rate_const_values` が `callable`、`has_time_dependent_rates` が真、種数が動的に変わる拡張、`numbalsoda_rhs` を生成できないその他の形。対象外は §4.3 または §3 の RK45 専用経路。
- **対象外の列挙（方針 A）**: 上記は**代表例**であり、網羅的な否定リスト（閉リスト）とはしない。本文ですべてを列挙する義務は負わない。例にない形も「生成できないその他」に含め §4.3／§3 に従う。README や改訂での補足は任意。
- 試行ごとに `p` の数値だけ更新し、`cfunc` の再コンパイルを避ける（§6.4）。

---

## 7. `numbalsoda.lsoda` の呼び出し

- `lsoda` は numbalsoda の公式 API に従う。
- **`atol`**: `solve_ode(..., atol=None)` の既定では `lsoda` に渡さない。`atol` が非 `None` のときのみ、公式 API が許す範囲で kwargs 等に渡す。rxnfit が LSODA 専用の独自 atol 既定を新設しない（§3.3 と整合）。
- **`lsoda` のその他のキーワード**（上記・§6.4 の `p` 等、本文で触れたもの以外）は numbalsoda の API・付属ドキュメントに従う。**本文では個別に列挙・既定値固定をしない**。`solve_ode` の公開シグネチャに無いオプションを利用者契約として保証しない。透過や列挙を仕様化する場合は改訂で行う。
- RHS が `p` を参照しないが API が `p` を要求する場合: **ゼロ長配列**または**ダミー配列**（未使用）のいずれか。どちらにするかは実装で選び、**実装コメントに一括で明記**する。

---

## 8. `solve_ivp` 戻り値と観測性

- `solve_ivp` が返す `message`・`success` 等は SciPy の値のまま。rxnfit は上書き・追記しない。
- 積分経路の分岐理由を **`warnings.warn` 以外**（`logging`、ファイル、標準出力の独自メッセージ等）で残さない。
- **観測性**: 利用者への通知は §4・§3.2 の `warnings.warn(..., UserWarning)` と §4.1 のキーに基づく初回抑止に限定する。**構造化ログ・テレメトリ・`logging` 等による分岐理由の別途記録は rxnfit 本体では採用しない**。

---

## 9. README 記載

新規セクション等で、少なくとも次を書く。

- optional dependency、`pip`/`conda` の requirement 差、`method="LSODA"` 時のみ numbalsoda 試行。
- numbalsoda／numba のバージョン列挙・動作確認済み例は書かない（§2）。
- §4.2 / §4.3 / §4.4 のフォールバックと英文の対応。§3.2 の時間依存英文・別キー。
- `method="LSODA"` 時の `t_eval`/`t_span`（§5 の適用範囲、`time_dependent=True` では §5 を適用しない旨を一文）。
- §3 の時間依存経路は `solve_ivp`+`RK45` のみ、`time_dependent` は呼び出し元が自動設定。警告の文面・キー分離。
- 警告キーは §4.1 の表どおり。
- 上級者: `p` の並びは §6.4 に従い、通常利用者は `p` を直接操作しない。
- 初版で LSODA が numbalsoda を試行し得る範囲は §6.5 に限定。

---

## 10. テストと CI

- テストツリーの詳細・ディレクトリ・網羅の拡充は本体実装が整ってから別途でもよい（早期に全文固定しない）。
- scipy vs numbalsoda の比較検証は **CI のみ**。ローカル通常利用では比較しない。
- `ci.yml` の作成・導入は実装 PR の付帯で同一 PR を目安にする。

**CI マトリクス（目安）**

- Linux: numbalsoda あり／なし（フォールバック）。
- Windows: `method="LSODA"` 時の RK45 フォールバック。
- macOS: Windows と同等のフォールバック。
- `pip` と `conda` を分けて検証可能なジョブ設計。

**テストデータ**: 既存 examples を利用可。CI 用は軽量サブセットを `tests/data` に。`tests` はリポジトリに含め、wheel 配布には含めない。

**最初の PR**: 最小の `ci.yml`（例: lint + numbalsoda なし pytest）でもよい。§10 の全マトリクスは後続 PR で拡張可。

**数値閾値（§11）**: 推奨値は文書化のみとし、CI ゲートの硬い数値は再現データが揃ってから固定する。

---

## 11. 数値閾値（同等性の目安）

- ハイブリッド判定: ODE 時系列は rtol/atol、フィット結果は rss / r2 / 推定パラメータ差。

**初期推奨値**

- 時系列: `rtol=1e-5`, `atol=1e-8`
- rss 相対差: `<= 1e-3`
- r2 絶対差: `<= 1e-4`
- 推定パラメータ相対差: `<= 5%`

---

## 12. 運用

- 本番は「利用可能なら numbalsoda、不可なら scipy」。
- 警告は `warnings.warn(..., UserWarning)`、同一理由は初回のみ（プロセス単位）。
- conda 経路の検証ジョブは任意。定期なら nightly を想定。

---

## 13. 将来拡張（一般）

- `pip` / `conda` の requirement 更新に追従。backend は import／実行可否ベース。OS 固定条件に依存しない。モデル構築・フィットへの影響は最小。

### 13.1 保留（テストツリー）

テストツリーの本格整備は §10 と整合させ別途。

---

## 14. `solve_ode` 直接利用者向け docstring

`solver_backend` を直接 import する利用者向けに、少なくとも次を docstring で明記する。

- `time_dependent=True` は §3 の時間依存速度定数経路（`rate_const_values` が `callable`、または `has_time_dependent_rates` が真）を呼び出し元が検出した結果として渡すこと。
- 既定は `False`。
- 誤って偽のまま呼んだ場合の挙動は保証しない（利用者責任）。

---

## 15. 実装 PR チェックリスト

- [ ] **本体**: `solver_backend` の経路・フォールバック。`method="LSODA"` のときのみ numbalsoda 試行。§4 のキー分離。§4.4 は `success=False` と例外でキー分割しない。§4.3 に JIT／コンパイル失敗を含め §4.4 と混同しない。`method != "LSODA"` は従来 `solve_ivp`。import キャッシュ。
- [ ] **§5・§5.5**: `build_t_eval_when_none_for_lsoda` / `LsodaImplicitTEvalMode` のみで `t_eval is None` を配列化。`solv_ode` / `expdata_fit` は LSODA かつ `None` のときだけ直前にヘルパー。numbalsoda 直前は非 `None` の `t_eval`。実効 `t_span` は LSODA 分岐内。`time_dependent=True` では §5 を適用しない。
- [ ] **§3**: `time_dependent` 自動判定、§3.2 警告、§3.3 の rtol/atol/t_eval/t_span。
- [ ] **`solve_ode` docstring**: §14。
- [ ] **観測性**: §8（`warnings` のみ）。
- [ ] **§7 末尾**: `p` 非参照時のゼロ長／ダミー選択を実装コメントで明記。
- [ ] **置換**: `solv_ode` / `expdata_fit` が `solve_ode` 経由。API 互換。
- [ ] **RHS**: `numbalsoda_rhs.py`、`create_system_rhs` からのハイブリッド。`numbalsoda_rhs` のみ `.address`。§6.5（対象外は例示・閉リストにしない）。`lsoda_p_vector.build_lsoda_parameter_vector_p`。§6.4 の docstring 三要素。§7 の atol・その他 kwargs・`p`。§8 の戻り値 `message`。手書き Numba に依存しない。
- [ ] **依存**: `pyproject.toml` の `[nlsoda]` optional、必須に含めない。
- [ ] **README**: §9。
- [ ] **テスト**: LSODA 試行、import 不可フォールバック、非 LSODA 維持、初回のみ警告、主要回帰。
- [ ] **CI**: `ci.yml`、Linux pip、Linux `.[nlsoda]` あり/なし、必須プラットフォーム、conda は任意。
- [ ] **最終**: ローカル pytest、CI green、本仕様と実装の一致（本書は `numbalsoda2.md` を正とする場合は参照を合わせる）。

---

*本書は `numbalsoda.md` の内容を再編した整理版である。*

# プロット機能の独立モジュール化（plot_results）

## 背景・目的

- `RxnODEsolver.solution_plot()` と `ExpDataFitSci.plot_fitted_solution()` は、いずれも時間経過プロットを提供している。
- 描画の実体はすでに `solv_ode._plot_time_course_solutions()` に集約されているが、この関数が `solv_ode` にあり、`expdata_fit_sci` が `solv_ode` に依存している。
- **目的**: プロット専用モジュール `plot_results.py` として独立させ、責務を分離する。より高機能な `plot_fitted_solution()` の仕様に合わせた共通 API に統一し、今後の他機能（感度解析・不確かさの可視化など）からも利用可能にする。

---

## 確定仕様

### 1. 公開 API

- プロット関数は **公開** する。
- 関数名: `plot_time_course_solutions`（先頭に `_` を付けない）。

### 2. サブプロットのタイトル（データセット名）

- 低レベル API に **optional** の `dataset_labels: Optional[List[str]]` を追加する。
- **名前が渡された場合**: 各サブプロットのタイトルにそのラベルを使用する。
- **名前が渡されていない場合（None または未指定）**: 現状どおり「Dataset 1」「Dataset 2」で表示する（B の動作）。

### 3. パッケージからの export

- **A**: `rxnfit/__init__.py` には追加しない。
- 利用時は `from rxnfit.plot_results import plot_time_course_solutions` のように **明示的に import** する。

### 4. 戻り値と表示の制御

- **戻り値**: `(fig, axes)` を返す（B）。
- **引数 `show`**: 表示の有無を切り替える。デフォルトは `True`（従来どおり `plt.show()` を呼ぶ）。

### 5. コンソール出力

- **A**: 現状の print（「=== Time-course plot ===」および最終時点の濃度の出力）を、そのままプロット関数内に含める。引数でのオン/オフは行わない。

### 6. 警告の stacklevel

- 積分失敗時の `warnings.warn(..., stacklevel=N)` について、**`stacklevel=4` に変更**する。
- 理由: 関数を `plot_results` に移すとコールスタックが 1 段増えるため、警告の表示位置を「ユーザーが `solution_plot()` / `plot_fitted_solution()` を呼んだ行」に合わせるには 4 とする必要がある。

### 7. モジュール名

- 新規モジュール名: **`plot_results.py`**
- 「結果（results）の可視化」であることが分かりやすく、略語の説明が不要であるためこの名前にする。

### 8. 型ヒントと docstring

- **A**: 引数・戻り値に型ヒントを付与し、docstring を新モジュール用に整える。

### 9. テスト

- **B**: 今回のリファクタではテストは追加しない。既存 example の手動確認のみとする。

---

## 実装により変更する .py ファイル一覧

| 種別 | ファイルパス | 変更内容 |
|------|----------------|----------|
| **新規** | `src/rxnfit/plot_results.py` | 新規作成。`plot_time_course_solutions()` を実装（`_plot_time_course_solutions` の移設＋仕様変更の反映）。`expdata_reader.get_time_unit_from_expdata` を import。引数に `dataset_labels`, `show` を追加。戻り値 `(fig, axes)`。`stacklevel=4`。型ヒント・docstring を整える。 |
| **変更** | `src/rxnfit/solv_ode.py` | `_plot_time_course_solutions` を削除。`plot_results` から `plot_time_course_solutions` を import。`solution_plot()` は解と df_list を組み立てたうえで `plot_time_course_solutions(...)` を呼ぶ薄いラッパーにし、戻り値は従来どおり返さない（またはそのまま返すかは実装時に合わせる）。 |
| **変更** | `src/rxnfit/expdata_fit_sci.py` | `_plot_time_course_solutions` の import 元を `solv_ode` から `plot_results` に変更（`plot_time_course_solutions` を import）。`plot_fitted_solution()` 内で、`df_names` からプロット対象分のラベルを切り出し、`dataset_labels` として `plot_time_course_solutions(...)` に渡す。戻り値の扱いは従来どおり（メソッドとしては返さない、またはそのまま返すかは実装時に合わせる）。 |

---

## 影響を受けないもの

- **examples（ノートブック）**: `solution_plot()` および `plot_fitted_solution()` のシグネチャを変えないため、修正不要。
- **README.md**: 必要に応じて「プロットは plot_results に集約」などの記述を追加可能（任意）。
- **その他モジュール**: `p0_opt_fit.py` などは `plot_fitted_solution` を直接呼ばないため、本リファクタの対象外。

---

## 参照

- 本仕様は、プロット機能の独立モジュール化に関する議論（solution_plot / plot_fitted_solution の統一、plot_results.py の新設）をまとめたものである。
- 上記「確定仕様」に従って実装する。

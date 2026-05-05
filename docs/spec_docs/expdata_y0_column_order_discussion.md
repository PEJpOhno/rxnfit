# 実験データの列順と初期値（y0）の設定 — 調査・方針・提言

## 0. プロンプト  

expdata_fit.pyでは、初期値として実験データの1行目を採用している。また、ヘッダ列には化学種の名称が記載されている。実験データの化学種の列の順番が、化学種のsortした順番と異なる場合、初期値をどのように設定しているか調査して

## 1. 調査の目的

- expdata_fit では初期値として**実験データの1行目**を採用している。
- ヘッダ列には**化学種の名称**が記載されている。
- **実験データの化学種の列の順番**が、**化学種の sort した順番**（または ODE 側の順番）と異なる場合、初期値がどのように設定されているかを把握する。

---

## 2. 調査結果（コードの流れ）

### 2.1 化学種の順番（ODE 側）`function_names`

- **定義場所**: `rxn_reader.py` の `get_unique_species(reaction_equations)`。
- **決まり方**: `sorted(set(flatten_species), key=flatten_species.index)` により、**反応式 CSV で化学種が初めて出現した順**で一意リストが作られる。
- **アルファベット sort ではない**。反応式の記述順（出現順）がそのまま `function_names` になる。
- この `function_names` が `build_ode` → `expdata_fit` に渡され、ODE の `y` の並び順として一貫して使われる。

### 2.2 実験データの列順

- **取得**: `expdata_read(df_list)` → `time_course(df)` で、`df.columns[1:]` の**そのままの並び**で `t_list`, `C_exp_list` が作られる。
- つまり実験 CSV の**ヘッダの並び順**＝データ側の列順。

### 2.3 初期値 y0 の設定（列順が異なる場合の挙動）

- **処理**: `expdata_fit.py` の `solve_fit_model_multi` 内で  
  `y0_list = get_y0_from_expdata(df_list, function_names)` を呼ぶ。
- **`get_y0_from_expdata`（expdata_reader.py）の実装**:
  - 各 DataFrame の **1行目** を `first_row = df.iloc[0]` で取得。
  - **`function_names` の順**でループ: `for name in function_names`。
  - 各 `name` について **列名で参照**: `val = first_row[name]`（`name in df.columns` を前提）。
  - 得た値をその順で `y0` に append。欠損は 0.0。

**結論**:  
初期値は**列の位置ではなく「化学種名（ヘッダ名）」で参照**している。  
したがって、**実験データの化学種の列の順番が、function_names（反応式での出現順）や「sort した順」と異なっていても、y0 は常に function_names の順で正しく設定される**。

### 2.4 経時データの整列

- **処理**: `align_expdata_to_function_names(t_list, C_exp_list, columns, function_names)`。
- `time_course` の出力（＝実験データの列順）を、**function_names の順**に並べ替える。
- `columns` は `df_list[0].columns[1:]`（実験データの化学種列名の並び）。各 `function_names` の name について `columns` 内のインデックスを求め、対応する `t_list[i]`, `C_exp_list[i]` を取る。
- その結果、残差計算に使う経時データも **function_names の順**で揃う。

---

## 3. まとめ（現状の仕様）

| 項目 | 内容 |
|------|------|
| 化学種の「順番」の定義 | 反応式 CSV での**出現順**（`get_unique_species`）。アルファベット sort ではない。 |
| 初期値 y0 | **1行目**の値を**列名（化学種名）で参照**して取得。常に **function_names の順**で格納。 |
| 実験データの列順が異なる場合 | 列順に依存せず、**名前で引く**ため y0 は正しく function_names 順になる。 |
| 経時データ | `align_expdata_to_function_names` で **function_names の順**に整列してから使用。 |

**「実験データの化学種の列の順番が、化学種の sort した順番と異なる場合」**:  
- 現状、**初期値は「名前」で取っているため、列順がどうであっても function_names の順で正しく設定されている**。  
- 「sort した順」は現状どこでも使っておらず、ODE の順序は一貫して **反応式での出現順（function_names）** のみ。

---

## 4. 今後のディスカッションの方向性・提言

- **仕様の明文化**: 「初期値は 1 行目を化学種名で参照し、ODE の化学種順（function_names）で保持する」ことを docstring や仕様書に書いておくと、列順を変えても安全であることが読み手に伝わる。
- **検証の推奨**: 実験データの列順を意図的に変えた CSV で、y0 とフィット結果が「列順を変える前」と一致することをテストで確認すると安心。
- **列名の不一致**: 現状は「化学種名がデータの列にない」場合は `ValueError`（`get_y0_from_expdata` および `align_expdata_to_function_names`）。大文字小文字や前後空白の違いで名前がずれるとエラーになるため、必要に応じて「列名の正規化・マッチング方針」を別途議論するとよい。
- **「sort した順」を採用するか**: 現状は採用していない。もし「実験データの列を常にアルファベット sort した順で解釈したい」などの要望が出た場合は、**どこで sort するか**（実験データ読み込み時のみか、ODE の function_names も sort するか）を仕様として決める必要がある。

---

*コードは変更していません。上記は調査と方針・提言の整理です。*

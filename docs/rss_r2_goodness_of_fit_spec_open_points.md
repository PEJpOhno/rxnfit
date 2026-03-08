# RSS / R² 併記（goodness_of_fit）仕様 — 決定事項と詳細

`rss` を廃止し、経時変化 vs 実験データに対して RSS と R² を返す API を統一する方針。以下に決定事項と、戻り値・run_fit まわりの詳細をまとめる。

---

## 決定済み項目

| # | 項目 | 決定内容 |
|---|------|----------|
| 1 | TSS の「平均」 | **化学種ごと**（後述の定義を採用） |
| 2 | 複数データセット時の R² | **全体で 1 つだけ**返す |
| 3 | 有効点の一致 | **明文化する**（後述の仕様文を採用） |
| 5 | TSS ≈ 0 のとき | **警告のみ**出し、R² は計算して返す |
| 6 | 重みづけ | R² は将来 **重み付き RSS/TSS** で定義する予定。**現状は重みなし**で仕様化 |
| 8 | goodness_of_fit のオプション | 現在の `rss` の **recompute をそのまま持たせる** |
| 9 | P0OptFit.optimize の戻り値 | **提案通り** `(out_dict, fit_metrics)`。fit_metrics は dict でキー `'rss'`, `'tss'`, `'r2'` を含む。 |

---

## 1. TSS の「平均」の定義（決定：化学種ごと）

- **採用する定義**  
  化学種 i ごとに、その種の有効観測値だけから平均 μ_i を計算する。  
  TSS = Σ_i Σ_{点∈種i} (C_obs − μ_i)²。  
  有効点は RSS を計算している点集合と完全に同一とする（複数データセットの場合は全データセットを合わせた有効点）。
- **全体で 1 つの R²** にするため、上記 TSS を全化学種・全データセットで合算した 1 つの TSS とし、R² = 1 − RSS/TSS で 1 つのスカラを返す。

---

## 2. 複数データセット時の R²（決定：全体で 1 つだけ）

- 複数 DataFrame を渡した場合も、RSS と同様に全データセットの有効観測で 1 つの TSS を計算し、**R² は 1 つのスカラのみ**返す。データセットごとの R² は返さない。

---

## 3. 有効点の定義の一致（決定：明文化する）

- **仕様として明文化する内容**  
  「RSS および TSS（したがって R²）は、**RSS を計算しているのと同じ有効点の集合だけ**を使って計算する。すなわち、時間または濃度が欠損（NaN）である (t, 化学種) は RSS にも TSS にも含めない。solv_ode と expdata_fit_sci のいずれでも、この有効点の定義を同じにする。」
- 実装時には、solv_ode 側（行・列の走査で NaN を skip）と expdata_fit_sci 側（種ごとの t_list[i], C_exp_list[i]）で、同じ実験データに対して同じ点集合が得られることを前提にし、上記を仕様・コメントに書いておく。

---

## 4. 戻り値の型 — 「何を決める話か」の詳細説明

ここで言う「戻り値の型」は、**RSS・TSS・R² を呼び出し側に返すとき、それらを「どういうデータ構造で渡すか」**を指しています。goodness_of_fit を呼んだあと、ユーザーが `rss` や `r2` にどうアクセスするかは、この型によって決まります。

### 4.1 何が「戻り値」になるか

- **solv_ode**  
  `goodness_of_fit(expdata_df)` は、解いた経時変化と実験データを比較した結果として、RSS・TSS・R² の 3 つ（以上）を返す必要があります。これらはまとめて 1 つの「戻り値」として返す想定です。
- **run_fit**  
  戻りは現状 `(result, param_info)` の 2 つ。RSS は `result.fun` で参照しています。R² を「返す」ということは、この 2 つのどちらか（または両方）に R² を載せるか、あるいは戻り値を増やすか、という話になります（これは 7 で扱う「格納先」に直結します）。
- **P0OptFit.optimize**  
  戻りは現状 `(out_dict, rss)` で、第二要素がスカラの RSS だけ。R² を追加するなら「第二要素を何にするか」が、まさに「戻り値の型」の話です。第二要素を「RSS と R² のまとまり」にするなら、そのまとまりを dict にするか NamedTuple にするか、が 4 の対象です。

つまり、**「RSS と R²（および必要なら TSS）を一まとまりで返すとき、そのまとまりを Python のどの型で表現するか」**が、項目 4 で扱っている内容です。

### 4.2 dict で返すとはどういう意味か

- **dict** は、キーと値の対応で複数の値を保持する型です。
- 例: `{"rss": 0.012, "tss": 0.45, "r2": 0.973}`。
- 呼び出し側は `metrics = solver.goodness_of_fit(df)` のあと、`metrics['rss']` や `metrics['r2']` でアクセスします。キー名を文字列で書くため、`'r2'` を `'r2'` と typo すると実行時まで気づかないことがあります。
- 一方で、新しい指標を足すときは `metrics['n_points'] = ...` のようにキーを増やすだけでよく、関数の戻り値の「形」を変えずに拡張できます。また、`metrics` をそのまま JSON にしたりログに書き出したりしやすい、という意味があります。

### 4.3 NamedTuple で返すとはどういう意味か

- **NamedTuple** は、名前付きの属性を持つタプルです。型を定義すると、各属性の名前が固定されます。
- 例: `GoodnessOfFit(rss=0.012, tss=0.45, r2=0.973)`。呼び出し側は `metrics.rss`、`metrics.r2` でアクセスします。属性名はドットで書くため、IDE の補完や型チェッカーの対象にしやすく、typo を減らせます。
- 一方で、あとから「r2_adj も返したい」となったときは、NamedTuple の定義に新しい属性を追加する必要があり、その型を使っているコードが型チェックで影響を受ける、という意味があります。

### 4.4 他 API との「形の揃い方」という意味

- **optimize** の第二戻り値を `(out_dict, {"rss": rss, "r2": r2})` にする、と決まっています。つまり「RSS と R² のまとまり」はすでに **dict** で返す形になっています。
- すると、goodness_of_fit の戻りも dict にすると、「RSS と R² の塊」を同じ型で扱えます。run_fit で R² を param_info に載せる場合、param_info はもともと dict なので、`param_info['r2']` のようにキーでアクセスすることになります。  
  つまり、**「まとまり」を dict にすると、optimize の第二戻り値・param_info の使い方と揃う**という意味があります。
- 逆に、goodness_of_fit だけ NamedTuple で返すと、optimize の第二戻り値は dict のままなので、「RSS・R² の塊」が API ごとに dict だったり NamedTuple だったりすることになります。揃えるなら、optimize の第二戻り値も NamedTuple にするか、goodness_of_fit も dict にするか、のどちらかになります。

### 4.5 まとめ（意味の説明のみ。仕様の決定はこのあと）

- 項目 4 は、**「RSS・TSS・R² を一まとまりで返すとき、そのまとまりを dict とするか NamedTuple とするか」**を決める話です。
- dict にすると、キーでアクセスし、拡張・シリアライズに有利で、optimize の第二戻り値や param_info と形を揃えやすいです。
- NamedTuple にすると、属性でアクセスし、型・補完・typo 防止に有利ですが、拡張時は型定義の変更が要り、他 API も同じ型に揃えるかどうかが問題になります。
- **どちらを採用するかは、上記の意味を踏まえたうえで、仕様として別途決定する。**

---

## 5. TSS が 0 に近い場合（決定：警告のみ）

- TSS が閾値 ε（例: 1e-10 や 1e-12）未満のときは、R² を **計算して返す**が、`warnings.warn(...)` で「TSS がほぼ 0 のため R² の解釈に注意」といった警告を出す。
- R² は `None` にせず float のまま返す（極端な値や inf になりうる場合は警告で補う）。

---

## 6. 重みづけ（決定）

- 将来、重み付き残差を導入する場合は、**R² は重み付き RSS と重み付き TSS で R²_w = 1 − RSS_w/TSS_w と定義する**方針とする。
- **現状は重みなし**で仕様化し、RSS / TSS / R² はいずれも重みなしで計算する。重み付きを入れる際は、同じ API で `weight` 引数などを追加し、RSS_w, TSS_w, R²_w を返す形に拡張する想定。

---

## 7. run_fit 内で R² を出すときの TSS 計算場所・戻り方 — 「何を決める話か」の詳細説明

ここでは、**run_fit（および run_fit_multi）のなかで R² を「計算して、呼び出し側に渡し、必要なら表示する」**ために、何を決める必要があるかを、用語の意味から説明します。仕様の決定はこの説明のあとで行う。

### 7.1 「run_fit 内で R² を出す」とは何を指しているか

run_fit で R² を「出す」には、次の 3 つが関わります。

1. **R² の計算**  
   R² = 1 − RSS/TSS なので、RSS と TSS を求める必要があります。RSS はすでに minimize の結果として `result.fun` で得ています。TSS は観測データだけから計算できます（モデルやパラメータに依存しない）。

2. **R² の格納・返し方**  
   計算した R² を「どこに入れて返すか」です。run_fit の戻りは現状 `(result, param_info)` なので、R² を返すには、このどちらかに載せるか、戻り値を増やすか、のいずれかになります。

3. **verbose での表示**  
   現在、run_fit は verbose が True のとき「Residual sum of squares: …」と RSS を表示しています。R² も併記するなら、そのときに表示する R² の値を、上記「格納先」のどこから取るか、が決まります。

項目 7 で言う「TSS 計算場所」は (1) の「どこで・どのタイミングで TSS を計算するか」、「戻り方」は (2)(3) の「R² をどこに格納し、verbose ではどこを参照するか」を指しています。

### 7.2 「TSS の計算場所」の意味

TSS は、fit_ctx に入っている `datasets`（各データセットの `t_list`, `C_exp_list`）から計算できます。有効な観測値だけを使い、化学種ごとの平均で TSS を求める、というのは決定済みです。

**「計算場所」が問題になるのは次の点です。**

- **どのタイミングで計算するか**  
  TSS はモデルに依存しないので、理論上は minimize の前でも後でも計算できます。一方、R² を出すには RSS（= result.fun）が必要なので、R² そのものは minimize の後に計算することになります。したがって、TSS も「minimize の後」に計算する、という流れが自然です。  
  「場所」というと、**run_fit のどの行のあたりで TSS を計算するか**（minimize が返った直後、param_info を組み立てる前か後か、など）という意味になります。

- **どの関数で計算するか**  
  TSS の計算ロジックを、run_fit のなかに直接書くか、それとも「datasets とオプション（化学種ごと平均など）を受け取って TSS を返す関数」として別に切り出すか、という意味です。  
  別関数にすると、単体テストしやすく、solv_ode 側の goodness_of_fit からも同じ関数を呼べる可能性があります。run_fit 内に直書きすると、流れは追いやすいが、他と共有するにはあとで関数に抽出する必要があります。

つまり、「TSS 計算場所」は **「run_fit のどのタイミングで TSS を計算するか」** と **「その計算を run_fit 内に書くか、共通関数として切り出すか」** の両方を含みます。

### 7.3 「R² の格納先」の意味

run_fit は `(result, param_info)` を返します。R² を「返す」ためには、この 2 つのどちらかに R² を入れるか、あるいは第三の戻り値を増やすか、です。

- **param_info に格納する**  
  `param_info` はもともと dict で、`symbolic_rate_consts` や `n_params` などのキーを持っています。ここに `'tss'` と `'r2'` を追加する、という意味です。  
  呼び出し側は `result, param_info = run_fit(...)` のあと、`param_info['r2']` で R² を参照します。RSS は従来どおり `result.fun` のままにしておくか、param_info に `'rss'` も重複して持たせるか、は別の設計の話です。

- **result に格納する**  
  `result` は scipy.optimize.minimize が返す `OptimizeResult` です。標準では `fun`, `x`, `success` などの属性があります。ここに、`result.r2 = ...` のように属性を追加する、という意味です。  
  呼び出し側は `result.r2` で R² を参照します。RSS は `result.fun` で、R² は `result.r2` で、と「結果オブジェクトに全部載せる」形になります。  
  ただし、OptimizeResult は scipy が返す型なので、そこに属性を足すことは「ライブラリの公式の使い方」ではなく、あくまでこのプロジェクト内の慣習になります。

- **戻り値を増やす**  
  `(result, param_info)` に加えて、第三戻り値として `metrics`（RSS, TSS, R² のまとまり）を返す、という形も考えられます。  
  呼び出し側は `result, param_info, metrics = run_fit(...)` のように受け、`metrics['r2']` などで参照します。戻り値の数と順序が変わるため、既存の `result, param_info = run_fit(...)` をしているコードは修正が必要になります。

「格納先」を決めるということは、**「R² を参照するとき、ユーザーは result と param_info のどちら（または別の戻り値）を見るか」** を決める、という意味です。それに合わせて、verbose で表示する R² も、同じ格納先から読むことになります。

### 7.4 verbose での表示の意味

現在、run_fit は verbose が True のとき、RSS を `result.fun` から取って表示しています。R² を併記するなら、同じタイミングで R² も print します。

**どこから R² を読むか**は、上記の格納先に依存します。param_info に R² を入れるなら `param_info['r2']` を表示し、result に R² を入れるなら `result.r2` を表示します。  
つまり、**「R² の格納先」を決めると、verbose の参照元も自動的に決まる**という関係にあります。

### 7.5 計算タイミングの流れ（意味の整理）

run_fit の処理の流れで言うと、次のようになります。

1. fit_ctx を組み立て、residual_func と param_info の初期内容を用意する。
2. minimize(residual_func, ...) を呼ぶ。
3. minimize が返ったあと、`result.fun` が RSS になる。
4. この時点で、fit_ctx['datasets'] から TSS を計算できる。
5. R² = 1 − result.fun / TSS を計算する。
6. 計算した TSS と R² を、選んだ「格納先」（param_info か result か）に書き込む。
7. verbose なら、RSS（result.fun）と R²（格納先から）を併記して表示する。
8. `(result, param_info)` を返す（格納先が param_info なら、その時点で param_info に r2 などが入っている）。

「TSS 計算場所」は、実質的には「ステップ 4 を、run_fit 内のどこで・どの関数で行うか」という意味になります。

### 7.6 まとめ（意味の説明のみ。仕様の決定はこのあと）

- 項目 7 は、**run_fit のなかで (1) TSS をいつ・どの関数で計算するか、(2) R²（と必要なら TSS）を result と param_info のどちらに格納して返すか、(3) verbose ではその格納先から R² を読んで表示するか**を決める話です。
- 「計算場所」は、minimize の後で TSS を計算することと、そのロジックを run_fit 内に書くか共通関数に切り出すか、を含みます。
- 「戻り方」は、R² を param_info に載せるか、result に属性として足すか、あるいは戻り値を増やすか、の選択であり、それに応じて呼び出し側の参照方法と verbose の参照元が決まります。
- **上記の意味を踏まえたうえで、仕様として別途決定する。**

---

## 8. goodness_of_fit のオプション（決定：recompute をそのまま持たせる）

- 新 API（goodness_of_fit）でも、現在の `rss` と同様に **recompute** 引数をそのまま持たせる。  
- recompute=True のときは実験時点で再積分して RSS を計算、recompute=False のときは既存 solution を補間して RSS を計算。TSS は実験データのみから求めるため recompute に依存しない。

---

## 9. P0OptFit.optimize の戻り値（決定・実装済み）

- 現在の `(out_dict, rss)` を **`(out_dict, fit_metrics)`** に変更した。
- 第二戻り値 **fit_metrics** は dict で、キー **'rss'**, **'tss'**, **'r2'** を含む。
- 共通モジュール **fit_metrics.py** の `fit_metrics(datasets, rss)` で算出し、run_fit の第三戻り値と同じ形で返す。

---

以上を仕様として実装に反映した。RSS と R² の意味・算出方法は一貫し、solv_ode（速度定数既知の経時変化）と expdata_fit_sci / P0OptFit（フィット後）の両方で同じ指標として扱える。

### 実装メモ（実施済み）

- **戻り値の型**: dict（キー `rss`, `tss`, `r2`）。
- **共通化**: 新モジュール **fit_metrics.py** に `fit_metrics(datasets, rss)` と `expdata_df_to_datasets(expdata_df, function_names)` を配置。TSS は化学種ごと平均。
- **run_fit / run_fit_multi**: 戻り値を **3 つ** `(result, param_info, fit_metrics)` に変更。result に属性 `tss`, `r2` を追加（RSS は result.fun のみ）。verbose は `fit_metrics['rss']` と `fit_metrics['r2']` を参照。
- **solv_ode**: 従来の `rss` を廃止し、**goodness_of_fit(expdata_df, ...)** メソッドで dict を返す。recompute オプションは維持。
- **P0OptFit.optimize**: 戻り値を `(out_dict, fit_metrics)` に変更。
- **TSS≈0 の警告**: 呼び出し側（run_fit 内・solv_ode の goodness_of_fit 内）で `warnings.warn` を実行。

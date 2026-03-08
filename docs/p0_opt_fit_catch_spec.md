# P0OptFit の catch 引数 仕様

## 概要

P0OptFit において、Optuna の `study.optimize(catch=...)` に渡す `catch` をデフォルトで設定し、コンストラクタの引数で変更できるようにする。

---

## (1) デフォルト値

- **デフォルトに含める例外**: `TrialPruned`, `RuntimeError`, `ValueError`
- **デフォルト**: `catch=(TrialPruned, RuntimeError, ValueError)`
- Optuna 標準の `(TrialPruned,)` に `RuntimeError` と `ValueError` を追加した形とする。

---

## (2) catch を渡す場所（パターン A）

- **コンストラクタのみ**で `catch` を指定する。
- `optimize()` の引数では変更できない。
- 実装方針:
  - `optimize()` 内で `study.optimize()` に渡すとき、`**kwargs` に `catch` が含まれていても渡さない（除外する）。
  - 常に `self._catch` を `study.optimize(..., catch=self._catch, ...)` のように渡す。

---

## (3) catch の型（パターン B）

- **tuple と list の両方**を受け付ける。
- 受け取った値は、保存時または `study.optimize()` に渡す前に **tuple に正規化**する（例: `tuple(catch)`）。
- Docstring では「例外クラスの列（tuple または list）」と記載する。

---

## (4) 空の catch および None の扱い（推奨どおり）

- **`catch=None`**
  - 許容する。
  - P0OptFit のデフォルト `(TrialPruned, RuntimeError, ValueError)` に変換して使用する。

- **`catch=()` または `catch=[]`**
  - 許容する。
  - 「いっさい追加で catch しない」＝厳格モードとして扱う。
  - `study.optimize(catch=())` に渡すと、目的関数がどの例外を出してもそのトライアルで最適化は止まる（伝播する）。
  - Docstring に「空の tuple/list を渡すと、目的関数のあらゆる例外が伝播し、そのトライアルで最適化が止まる」と明記する。

---

## 変更対象モジュール

- **Python コード**: `src/rxnfit/p0_opt_fit.py` のみを変更する。
- ドキュメント: 必要に応じて `docs/p0_opt_fit.md` 等に `catch` の説明を追記する。

---

## 仕様まとめ一覧

| 項目 | 内容 |
|------|------|
| デフォルト値 | `(TrialPruned, RuntimeError, ValueError)` |
| 指定場所 | コンストラクタのみ（`optimize()` では変更不可） |
| 型 | tuple または list を受け付け、内部で tuple に正規化 |
| `None` | デフォルトに変換 |
| 空 `()` / `[]` | 許容。catch しない（厳格モード） |

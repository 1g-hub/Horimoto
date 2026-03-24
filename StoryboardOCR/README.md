# Storyboard code (VLM + CLM)

Anime storyboard (絵コンテ) を対象に, OCR・補正・シーン説明を段階的に実行し, 説明可能な構造化データへ変換するための研究用コード.

- Stage0: 脚本などの外部知識の抽出・整形
- Stage1: VLM による列別 OCR（Base / LoRA / Mixed）
- Stage2: CLM による OCR テキスト補正（外部知識を根拠に補正）
- Stage3: 画像 + 補正済みテキストの統合解釈によるシーン説明生成
- Evaluation: CER / EM / 矢印指標などの評価

> **Note**
> 本リポジトリは研究・再現用. 絵コンテ画像・脚本などの一次データは著作権の都合で同梱しない想定.

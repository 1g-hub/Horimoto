# Storyboard Understanding Pipeline (VLM + CLM)

Anime storyboard (絵コンテ) を対象に、OCR・補正・シーン説明を段階的に実行し、説明可能な構造化データへ変換するための研究用コード群。

- Stage0: 脚本などの外部知識の抽出・整形
- Stage1: VLM による列別 OCR（Base / LoRA / Mixed）
- Stage2: CLM による OCR テキスト補正（外部知識を根拠に補正）
- Stage3: 画像 + 補正済みテキストの統合解釈によるシーン説明生成
- Evaluation: CER / EM / 矢印指標などの評価

> **Note**
> 本リポジトリは研究・再現用。絵コンテ画像・脚本などの一次データは著作権の都合で同梱しない想定。


---

## 1. Environment

- OS: Ubuntu (推奨)
- Python: 3.10+（推奨）
- GPU: NVIDIA GPU + CUDA（VLM 推論・学習を行う場合）
- Main libraries:
  - `torch`
  - `transformers`
  - `peft`（LoRA）
  - `Pillow`
  - `tqdm`
  - `matplotlib`
  - `numpy`

> **Tip**
> `torch` は環境によりインストール手順が異なるため、公式の案内に従うことを推奨。


---

## 2. Install (pip)

### 2.1 venv 作成

python -m venv .venv
source .venv/bin/activate
pip install -U pip
2.2 依存ライブラリ導入
pip install -r requirements.txt
requirements.txt を使わない場合の例:
pip install torch transformers peft pillow tqdm matplotlib numpy
3. Dataset / Input Format
本パイプラインは「ページ画像」および「列 crop（cut / picture / action_memo / dialogue / time）」を前提にする。
3.1 主要ファイル
data/episode01/annotation_table.csv
GT（final_gt）を含むアノテーションテーブル
主に以下の列を利用
cut,page,row,column,final_gt,...
data/episode01/<column>/page{page}_row{row}.png
列 crop 画像
<column> は cut, picture, action_memo, dialogue, time
data/arrow_crops_*.csv（任意）
矢印切り抜き（arrow crop）用データセット
矢印学習（LoRA）を強化する目的
4. Directory Overview
代表的なディレクトリ構成の例。
data/
  episode01/
    pages/                  # PDF から作成したページ画像
    cut/ picture/ action_memo/ dialogue/ time/
                            # 列 crop 画像（pageX_rowY.png）
    annotation_table.csv
  arrows/                   # 矢印 crop 画像（任意）
  arrow_crops_*.csv          # 矢印 crop のメタ情報（任意）

outputs/
  episode01/
    cuts/                    # Stage1 の cut 単位 OCR 結果（cutXXXX.stage1.json）

outputs_ft/
  stage1_ocr_lora_*/         # Stage1 LoRA 学習結果
    adapter/                 # LoRA adapter
    split_by_cut.json        # cut 分割（train/val/test）
    column_model_map.json    # Mixed 用（列→base/lora）
    eval_test_compare.json   # base vs lora vs mixed の評価

outputs_stage2_clm/
  episode01/
    cuts_split/              # Stage2 出力（cutXXXX.stage2.split.{nojp|jp}.json）
    eval_*/                  # Stage2 評価結果（json / jsonl）
5. Pipeline Usage (Stage0 → Stage3)
下記は「実行手順の雛形」。
実際のスクリプト名・引数はリポジトリ内のファイル名に合わせて置換すること。
5.1 (Optional) PDF → ページ画像
目的: 絵コンテ PDF をページ画像へ分割。
python tools/pdf_to_images.py \
  --pdf data/episode01/storyboard.pdf \
  --out data/episode01/pages
5.2 (Optional) ページ画像 → 列 crop 生成
目的: 絵コンテのレイアウトに合わせて列ごとの画像を生成。
python tools/crop_storyboard_columns.py \
  --pages_dir data/episode01/pages \
  --out_dir data/episode01
出力先例:
data/episode01/cut/
data/episode01/picture/
data/episode01/action_memo/
data/episode01/dialogue/
data/episode01/time/
5.3 Stage0: 外部知識の抽出（脚本・用語辞書など）
目的: Stage2 で利用する「根拠情報」を整形・保存。
python stage0_extract_knowledge.py \
  --script_path data/episode01/script.txt \
  --out_dir data/episode01/knowledge
想定出力:
登場人物辞書
制作用語辞書
セリフ候補一覧（検索用）
5.4 Stage1: VLM OCR（Base / LoRA / Mixed）
目的: 列 crop 画像から OCR を行い、cut 単位の JSON を出力。
python stage1_ocr.py \
  --episode_id episode01 \
  --episode_dir data/episode01 \
  --out_dir outputs/episode01/cuts \
  --mode mixed \
  --column_model_map outputs_ft/stage1_ocr_lora_*/column_model_map.json
出力例:
outputs/episode01/cuts/cut0001.stage1.json
5.5 Stage2: CLM による OCR 補正
目的: Stage1 出力を、脚本・辞書などを根拠に補正。
python stage2_clm_correct.py \
  --episode_id episode01 \
  --stage1_dir outputs/episode01/cuts \
  --knowledge_dir data/episode01/knowledge \
  --out_dir outputs_stage2_clm/episode01/cuts_split \
  --variant nojp
出力例:
outputs_stage2_clm/episode01/cuts_split/cut0001.stage2.split.nojp.json
outputs_stage2_clm/episode01/cuts_split/cut0001.stage2.split.jp.json
5.6 Stage3: シーン説明生成（画像 + 補正済みテキスト）
目的: cut 単位に、演出意図を含む説明文や構造化データを生成。
python stage3_scene_description.py \
  --episode_id episode01 \
  --stage2_dir outputs_stage2_clm/episode01/cuts_split \
  --out_dir outputs_stage3/episode01

6. Evaluation
6.1 Stage1 (Base vs LoRA vs Mixed) の評価

目的: split_by_cut.json の test_cut_ids に基づいて評価。

主な指標:

CER, Exact Match

矢印指標（Presence / Count / Direction / Sequence など）

python eval_compare_base_vs_lora_test_plus_arrows_bycol.py


出力例:

outputs_ft/.../eval_compare_arrows/compare_base_vs_lora_test_with_arrows.json

outputs_ft/.../eval_compare_arrows/*.png

outputs_ft/.../eval_compare_arrows/*.jsonl

6.2 Stage2 (補正前後) の評価

目的: GT（annotation_table.csv）と、Stage1 / Stage2 を比較。

Stage2 の出力が欠損している cut / cell を評価から除外せず、
Stage2 = Stage1 とみなして評価に含める設定のスクリプトを使用可能。

python eval_stage2_vs_gt_split_testcuts_and_all_include_failures.py


出力例:

outputs_stage2_clm/episode01/eval_*/eval_stage2_vs_gt_*.json

outputs_stage2_clm/episode01/eval_*/samples_*.jsonl

7. Stage1 JSON Format (example)

outputs/episode01/cuts/cut0001.stage1.json は cut 単位で、複数 row を持つ。

rows[].cols.<column>.raw_text を OCR 結果として利用

model_variant に base / lora が入る場合がある（Mixed 戦略）

8. Reproducibility Notes

split は split_by_cut.json に保存され、同一 test_cut_ids で再評価可能。

Stage2 の失敗（cut ファイル欠損 / cell 欠損）は、
Stage2 = Stage1 として評価に含める実装を採用可能。

9. License / Citation

License: (TBD)

Citation: (TBD)
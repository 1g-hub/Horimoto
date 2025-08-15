# このスクリプトは、「AnimeList.csv」を元にWikipediaから関連する記事を取得し、
# その内容を検索しやすいように小さなチャンクに分割します。
# 最終的に、データベース投入用の中間ファイル「prepared_anime_chunks.jsonl」を生成します。
# この処理ではLLMを使用しないため、比較的CPUベースで高速に動作します。

import pandas as pd
import wikipedia
import time
import re
import unicodedata
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# --- 設定項目 ---
CSV_FILE_PATH = "AnimeList.csv"
OUTPUT_FILE = "prepared_anime_chunks.jsonl"
PROGRESS_FILE = "prepare_progress.log"
CHUNK_SIZE = 1000  # 各チャンクの最大文字数
CHUNK_OVERLAP = 100   # チャンク間の重複文字数
wikipedia.set_lang("ja")

# ==============================================================================
# Wikipedia高精度検索モジュール (変更なし)
# ==============================================================================
def generate_search_candidates(title):
    if not isinstance(title, str): return []
    normalized_title = unicodedata.normalize('NFKC', title)
    candidates = {normalized_title}
    base_title = re.sub(r'\(.*?\)|\[.*?\]|【.*?】', '', normalized_title).strip()
    if base_title and base_title != normalized_title: candidates.add(base_title)
    parts = [p for p in re.split(r'[\s\-–—:/・]', base_title) if p]
    current_candidate = ""
    for part in parts:
        current_candidate = f"{current_candidate} {part}".strip()
        candidates.add(current_candidate)
    sorted_candidates = sorted(list(candidates), key=len, reverse=True)
    final_candidates = []
    for cand in sorted_candidates:
        final_candidates.extend([cand, f"{cand} (アニメ)", f"{cand} (漫画)"])
    return list(dict.fromkeys(final_candidates))

def search_wikipedia_robust(title):
    candidates = generate_search_candidates(title)
    if not candidates: return None
    for query in candidates:
        try:
            page = wikipedia.page(query, auto_suggest=False, redirect=True)
            anime_keywords = ["アニメ", "漫画", "ライトノベル", "ゲーム"]
            if any(keyword in category for category in page.categories for keyword in anime_keywords):
                print(f"  -> ✅ ヒット: '{title}' -> '{page.title}' (クエリ: '{query}')")
                return page
        except Exception:
            continue
    print(f"  -> ❌ 記事が見つかりませんでした: '{title}'")
    return None

# ==============================================================================
# メイン処理
# ==============================================================================
def load_processed_ids():
    if not os.path.exists(PROGRESS_FILE): return set()
    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}

def prepare_chunks_from_csv():
    processed_ids = load_processed_ids()
    print(f"\n{len(processed_ids)}件は処理済み。続きから開始します。")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"エラー: CSVファイル '{CSV_FILE_PATH}' が見つかりません。")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_fh, \
         open(PROGRESS_FILE, 'a', encoding='utf-8') as prog_fh:
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing Chunks"):
            anime_id = str(row.get("anime_id", ""))
            if not anime_id or anime_id in processed_ids:
                continue

            title_to_search = row.get('title_japanese', row.get('title'))
            page = search_wikipedia_robust(title_to_search)

            if page:
                # 記事コンテンツをチャンクに分割
                chunks = text_splitter.split_text(page.content)
                
                for i, chunk_text in enumerate(chunks):
                    # 各チャンクに必要な情報を付加してJSONオブジェクトを作成
                    chunk_data = {
                        "anime_id": int(anime_id),
                        "title_japanese": row.get("title_japanese"),
                        "wiki_title": page.title,
                        "wiki_url": page.url,
                        "chunk_id": i + 1,
                        "chunk_text": chunk_text,
                    }
                    out_fh.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
            
            # 処理済みIDを記録
            prog_fh.write(anime_id + '\n')
            prog_fh.flush()
            out_fh.flush()
            time.sleep(0.5) # APIへの負荷軽減

if __name__ == "__main__":
    prepare_chunks_from_csv()
    print(f"\n全ての処理が完了しました。中間ファイルは '{OUTPUT_FILE}' を確認してください。")
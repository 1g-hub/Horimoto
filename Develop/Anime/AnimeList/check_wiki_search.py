# このスクリプトは、MyAnimeListのデータセット(CSV)を読み込み、
# 各アニメの日本語タイトルを使ってWikipediaを検索します。
# 検索結果と元のタイトルを比較表示することで、Wikipedia検索の
# マッチング精度を人間が視覚的に確認することを目的とします。

import pandas as pd
import wikipedia
import time
import re
import unicodedata

# --- 設定項目 ---
CSV_FILE_PATH = "AnimeList.csv"
# Wikipediaの言語を日本語に設定
wikipedia.set_lang("ja")


def generate_search_candidates(title):
    """
     một tiêu đề, tạo ra một danh sách các ứng viên tìm kiếm có khả năng.
    Ví dụ: '灼眼のシャナII –Second–' -> ['灼眼のシャナII –Second–', '灼眼のシャナII', '灼眼のシャナ']
    """
    if not isinstance(title, str):
        return []

    # 1. Unicode正規化 (NFKC) を行い、全角記号などを半角に
    normalized_title = unicodedata.normalize('NFKC', title)

    # 候補リストの初期化（元の正規化タイトルも候補に入れる）
    candidates = {normalized_title}

    # 2. 括弧とその中身を削除したバージョンを候補に追加
    base_title = re.sub(r'\(.*?\)|\[.*?\]|【.*?】', '', normalized_title).strip()
    if base_title and base_title != normalized_title:
        candidates.add(base_title)
    
    # 3. 区切り文字で分割し、前方から部分文字列を生成
    #    区切り文字: スペース、ハイフン類、コロン、スラッシュなど
    #    例: '灼眼のシャナ II Second'
    #    -> ['灼眼のシャナ', '灼眼のシャナ II', '灼眼のシャナ II Second']
    #
    #    'BLEACH - ブリーチ -'
    #    -> ['BLEACH', 'BLEACH -', 'BLEACH - ブリーチ', 'BLEACH - ブリーチ -']
    
    # 複数の区切り文字を正規表現で指定
    delimiters = r'[\s\-–—:/・]'
    parts = re.split(delimiters, base_title)
    
    # 空の要素を除去
    parts = [p for p in parts if p]
    
    # 前方から結合して候補を生成
    current_candidate = ""
    for part in parts:
        # 最初の要素はそのまま
        if not current_candidate:
            current_candidate = part
        else:
            # 2つ目以降はスペースを挟んで結合
            current_candidate += " " + part
        
        candidates.add(current_candidate.strip())

    # 4. 長い順にソートして、より具体的な候補から試せるようにする
    #    重複を除去するために一度setにしてからlistに戻す
    sorted_candidates = sorted(list(candidates), key=len, reverse=True)
    
    # アニメ検索用のサフィックスを追加した候補も生成
    final_candidates = []
    for cand in sorted_candidates:
        final_candidates.append(cand)
        final_candidates.append(f"{cand} (アニメ)")
        final_candidates.append(f"{cand} (漫画)")
        
    # 最終的なリストから重複を再度除去して返す
    return list(dict.fromkeys(final_candidates))


def search_wikipedia_robust(title):
    """
    候補生成関数を使い、最も確実な検索を行う関数
    """
    # 1. 検索候補リストを生成
    candidates = generate_search_candidates(title)
    if not candidates:
        return {"status": "Skipped (No Candidates)"}

    # 2. 候補を順番に試す
    for query in candidates:
        try:
            page = wikipedia.page(query, auto_suggest=False, redirect=True)
            
            # 3. カテゴリ検証（重要）
            anime_keywords = ["アニメ", "漫画", "ライトノベル", "ゲーム"]
            if any(keyword in category for category in page.categories for keyword in anime_keywords):
                # 成功！
                return {
                    "status": "Success",
                    "candidates": candidates,
                    "query": query,
                    "found_title": page.title,
                    "summary": page.summary.split('\n')[0],
                    "url": page.url
                }
            # カテゴリが不適切なら、この結果は無視して次の候補へ
            
        except wikipedia.exceptions.PageError:
            continue # ページがなければ次の候補へ
        except wikipedia.exceptions.DisambiguationError:
            # 曖昧さ回避は、今のところスキップして次の候補へ（よりシンプルにするため）
            continue
        except Exception:
            # ネットワークエラーなど
            continue
            
    # 全ての候補で失敗した場合
    return {"status": f"Failed (No relevant page found for '{title}', '{candidates}')"}


if __name__ == "__main__":
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"エラー: CSVファイル '{CSV_FILE_PATH}' が見つかりません。")
        exit()

    print("--- Wikipedia検索精度チェック開始 ---")
    
    for index, row in df[16:18].iterrows():
        original_title_jp = row['title_japanese']
        
        print("\n" + "="*50)
        print(f"[{index+1}] 検索対象: {row['title']} ({original_title_jp})")
        print("="*50)
        
        result = search_wikipedia_robust(original_title_jp)
        
        print(f"  [検索結果]")
        print(f"    ステータス: {result['status']}")
        if result['status'] == 'Success':
            print(f"    候補: '{result['candidates']}'")
            print(f"    実行クエリ: '{result['query']}'")
            print(f"    ヒットしたタイトル: {result['found_title']}")
            print(f"    URL: {result['url']}")
            print(f"    概要: {result['summary']}")
            # 元のタイトルとヒットしたタイトルが一致しているか簡易チェック
            if original_title_jp in result['found_title'] or result['found_title'] in original_title_jp:
                print("    評価: ✅ マッチング成功の可能性が高い")
            else:
                print("    評価: ⚠️ マッチングが不正確な可能性あり")
        elif result['status'] == 'Failed (Disambiguation)':
            print(f"    候補: {result['options']}")
            print("    評価: ❌ 曖昧さの解決が必要")
        else:
             print("    評価: ❌ 検索失敗")
        
        # APIへの負荷軽減
        time.sleep(1)

    print("\n--- チェック完了 ---")
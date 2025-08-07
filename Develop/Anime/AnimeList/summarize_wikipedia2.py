# このスクリプトは、指定されたキーワードでWikipediaの記事を取得し、
# ローカルLLMを使って「あらすじ」「登場人物」「その他」の3つの観点で
# 構造化された要約を生成します。

import torch
import wikipedia

from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# --- 設定項目 ---
SEARCH_TERM = "クレヨンしんちゃん"
model_name = "elyza/Llama-3-ELYZA-JP-8B"

# Wikipediaの言語設定
wikipedia.set_lang("ja")

def get_wikipedia_content(term):
    """Wikipediaから記事の全文コンテンツを取得する"""
    try:
        print(f"Wikipediaで「{term}」を検索中...")
        page = wikipedia.page(term, auto_suggest=False, redirect=True)
        print(f"記事が見つかりました: {page.title} ({page.url})")
        return page.content
    except Exception as e:
        print(f"記事の取得に失敗しました: {e}")
        return None
    


def create_structured_chunk_summary(article_content):

    map_prompt = """
        あなたは優秀な編集者です。特に指示が無い場合は、常に日本語で回答してください。
        あなたのタスクは、与えられた記事を、後で他の部分と結合して最終的な要約を作成するために、以下の指示にしたがって要点を簡潔にまとめることです。

        # 指示:
        1.  記事全体を注意深く読み、以下の3つのセクションに情報を分類・整理してください。
            - **あらすじ**: 物語の基本的な設定やストーリーの概要を記述します。
            - **登場人物**: 主要なキャラクターやその関係性を中心に記述します。
            - **その他**: 作品の社会的影響、メディア展開（アニメ、映画、ゲームなど）、特筆すべき制作背景や豆知識など、上記2つに含まれない重要な情報を記述します。
        2.  各セクションは、見出し（例: `### あらすじ`）から始めてください。
        3.  要約は、元の記事の情報を忠実に反映し、あなたの意見や外部知識は含めないでください。
        4.  平易で分かりやすい言葉を使って、簡潔にまとめてください。

        # 出力形式:
        ### あらすじ
        （ここに要約を記述）

        ### 登場人物
        （ここに要約を記述）

        ### その他
        （ここに要約を記述）
        """
    
    map_text = f"{article_content}\n\n上記のWikipediaの記事を読んで、指定された3つの観点から内容を要約し、構造化された形式で出力してください。"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": map_prompt},
        {"role": "user", "content": map_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
    chunk_summary = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
    )

    return chunk_summary


def create_structured_summary(chunk_summary):

    combine_prompt = f"""
        あなたは優秀な編集者です。特に指示が無い場合は、常に日本語で回答してください。
        あなたのタスクは、ある記事の各部分を要約したものを以下の指示にしたがって結合することです。

        # 指示:
        1.  記事全体を注意深く読み、以下の3つのセクションに情報を分類・整理してください。
            - **あらすじ**: 物語の基本的な設定やストーリーの概要を記述します。
            - **登場人物**: 主要なキャラクターやその関係性を中心に記述します。
            - **その他**: 作品の社会的影響、メディア展開（アニメ、映画、ゲームなど）、特筆すべき制作背景や豆知識など、上記2つに含まれない重要な情報を記述します。
        2.  各セクションは、見出し（例: `### あらすじ`）から始めてください。
        3.  要約は、元の記事の情報を忠実に反映し、あなたの意見や外部知識は含めないでください。
        4.  平易で分かりやすい言葉を使って、簡潔にまとめてください。

        # 出力形式:
        ### あらすじ
        （ここに要約を記述）

        ### 登場人物
        （ここに要約を記述）

        ### その他
        （ここに要約を記述）
        """
    
    combine_text = f"{chunk_summary}\n\n上記のWikipediaの記事の各部分を要約したものを読んで、指定された3つの観点から内容を要約し、構造化された形式で出力してください。"


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": combine_prompt},
        {"role": "user", "content": combine_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
    summary = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
    )

    return summary


if __name__ == "__main__":

    # 保存先のファイル名
    filename = "output.txt"
    
    # Wikipediaから記事内容を取得
    content = get_wikipedia_content(SEARCH_TERM)
    
    if content:

        # テキストをチャンクに分割
        # chunk_sizeは、モデルのトークン限界とプロンプトの長さを考慮して決める
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = [Document(page_content=x) for x in text_splitter.split_text(content)]
        print(f"記事を {len(docs)} 個のチャンクに分割しました。")

        with open(filename, 'a', encoding='utf-8') as f:
            chunk_summary = []
            for document in docs:
                output = create_structured_chunk_summary(document)
                content_to_save = f"\n【wikipedia 記事】:\n{document}\n【要約】:\n{output}"
                f.write(content_to_save)
                chunk_summary.append(output)
            
        chunk_summary = '\n'.join(chunk_summary)
        summary = create_structured_summary(chunk_summary)
        
        # 結果の表示
        print("\n" + "="*50)
        print("【生成された要約】")
        print("="*50)
        print(summary)

        with open(filename, 'a', encoding='utf-8') as f:
            # ファイルに文字列を書き込む
            content_to_save = f"\n【最終的な要約】:\n{summary}"
            f.write(content_to_save)
        print(f"結果を '{filename}' に保存しました。")
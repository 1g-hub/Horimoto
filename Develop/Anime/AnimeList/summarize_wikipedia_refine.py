# このスクリプトは、指定されたキーワードでWikipediaの記事を取得し、
# ローカルLLMを使って「Refine（改良）」方式で構造化された要約を生成します。
# チャンクを一つずつ処理し、前の要約を更新していくことで、最終的な要約を構築します。

import torch
import wikipedia

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 設定項目 ---
SEARCH_TERM = "クレヨンしんちゃん"
MODEL_NAME = "elyza/Llama-3-ELYZA-JP-8B"
OUTPUT_FILENAME = "output_refine.txt"

# Wikipediaの言語設定
wikipedia.set_lang("ja")

def get_wikipedia_content(term):
    try:
        print(f"Wikipediaで「{term}」を検索中...")
        page = wikipedia.page(term, auto_suggest=False, redirect=True)
        print(f"記事が見つかりました: {page.title} ({page.url})")
        return page.content
    except Exception as e:
        print(f"記事の取得に失敗しました: {e}")
        return None

def initialize_model_and_tokenizer(model_name):
    """モデルとトークナイザーを一度だけロードする"""
    print(f"Tokenizer for '{model_name}' をロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Model '{model_name}' をロード中...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def run_llm_generation(model, tokenizer, messages, max_tokens=1200):
    """LLMの生成処理を共通化"""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id # pad_token_idを設定
        )
    
    response_text = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
    )
    return response_text

# --- Refine方式のための関数 ---

def create_initial_summary(document, model, tokenizer):
    """最初のチャンクから、暫定的な要約を作成する"""
    system_prompt = """
    あなたは優秀な編集者です。与えられた記事の最初の部分を読み、以下の指示に従って要約を作成してください。
    この要約は後で新しい情報を使って更新されます。

    # 指示:
    1.  以下の3つのセクションに情報を分類・整理してください。
        - あらすじ: 物語の基本的な設定やストーリーの概要。
        - 登場人物: 主要なキャラクターやその関係性。
        - その他: 上記以外で特筆すべき情報。
    2.  各セクションは見出し（例: `### あらすじ`）から始めてください。
    3.  記事に情報がないセクションは作成しないでください。
    """
    
    user_prompt = f"""
    以下のテキストを要約してください。

    # テキスト:
    {document.page_content}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return run_llm_generation(model, tokenizer, messages)

def refine_summary(existing_summary, document, model, tokenizer):
    """既存の要約を、新しいドキュメントの情報を使って更新・改良する"""
    system_prompt = """
    あなたは優秀な編集者です。あなたのタスクは、既存の要約を、新しく与えられたテキストの情報を使って更新し、より完璧なものにすることです。

    # 指示:
    1.  既存の要約と新しいテキストの両方を注意深く読んでください。
    2.  新しいテキストに含まれる情報を、既存の要約の適切なセクション（あらすじ、登場人物、その他）に追加・統合してください。
    3.  情報の重複を避け、全体として一貫性のある、より詳細で正確な要約を作成してください。
    4.  出力は、必ず「あらすじ」「登場人物」「その他」のセクション構造を維持してください。
    """

    user_prompt = f"""
    # 既存の要約:
    {existing_summary}

    # 新しいテキスト:
    {document.page_content}

    上記のルールに従って、既存の要約を新しいテキストの情報で更新してください。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return run_llm_generation(model, tokenizer, messages)

# --- メイン処理 ---
if __name__ == "__main__":
    content = get_wikipedia_content(SEARCH_TERM)
    
    if content:
        model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

        # テキストをチャンクに分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = [Document(page_content=x) for x in text_splitter.split_text(content)]
        print(f"記事を {len(docs)} 個のチャンクに分割しました。")

        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(f"# {SEARCH_TERM} の構造化要約 (Refine方式)\n\n")
            
            # Refineプロセスの開始
            print("\n--- Refineプロセスを開始します ---")
            
            # 1. 最初のチャンクで初期要約を作成
            print(f"[1/{len(docs)}] 最初のチャンクで初期要約を作成中...")
            current_summary = create_initial_summary(docs[0], model, tokenizer)
            f.write(f"\n\n【wikipedia 記事】:\n{docs[0]}\n\n【初期要約】:\n{current_summary}")
            print("初期要約が作成されました。")

            # 2. 2番目以降のチャンクで要約を更新していく
            for i, doc in enumerate(docs[1:], start=2):
                print(f"[{i}/{len(docs)}] 要約を更新中...")
                current_summary = refine_summary(current_summary, doc, model, tokenizer)
                f.write(f"\n\n[{i}/{len(docs)}] 要約を更新中...\n【wikipedia 記事】:\n{docs[0]}\n\n【要約】:\n{current_summary}")
                print(f"チャンク {i} の情報で要約が更新されました。")

            # 最終的な要約の表示と保存
            print("\n" + "="*50)
            print("【最終的に生成された要約】")
            print("="*50)
            print(current_summary)
            print(f"\n結果を '{OUTPUT_FILENAME}' に保存しました。")
# このスクリプトは、指定されたキーワードでWikipediaの記事を取得し、
# ローカルLLMを使って「あらすじ」「登場人物」「その他」の3つの観点で
# 構造化された要約を生成します。

import wikipedia
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 設定項目 ---
SEARCH_TERM = "クレヨンしんちゃん"
LLM_MODEL = "elyza/Llama-3-ELYZA-JP-8B"

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

def initialize_llm():
    """LLMパイプラインを初期化する"""
    print(f"LLM「{LLM_MODEL}」をロード中... (初回は時間がかかります)")

    # 1. トークナイザーをロードし、設定を変更する
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. モデル本体をロードする
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 3. Hugging Faceの標準的な`pipeline`を自分で作成する
    #    この時に、カスタマイズしたトークナイザーとロードしたモデルを渡す
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # パイプラインの引数もここで指定
        temperature=0.1,
        repetition_penalty=1.1
    )

    # 4. 作成した標準パイプラインを、LangChainのラッパークラスに渡す
    #    この方法なら、余分な引数として扱われず、エラーにならない
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    return llm



def create_structured_summary(llm_pipeline, article_content):
    """LLMを使って構造化された要約を生成する"""
    
    # LLMへの指示を詳細に記述したプロンプトテンプレート
    # このプロンプトが要約の品質を大きく左右します。
    template = """
    あなたは優秀な編集者です。以下のWikipediaの記事を読んで、指定された3つの観点から内容を要約し、構造化された形式で出力してください。

    # 指示:
    1.  記事全体を注意深く読み、以下の3つのセクションに情報を分類・整理してください。
        - **あらすじ**: 物語の基本的な設定やストーリーの概要を記述します。
        - **登場人物**: 主要なキャラクターやその関係性を中心に記述します。
        - **その他**: 作品の社会的影響、メディア展開（アニメ、映画、ゲームなど）、特筆すべき制作背景や豆知識など、上記2つに含まれない重要な情報を記述します。
    2.  各セクションは、見出し（例: `### あらすじ`）から始めてください。
    3.  要約は、元の記事の情報を忠実に反映し、あなたの意見や外部知識は含めないでください。
    4.  平易で分かりやすい言葉を使って、簡潔にまとめてください。

    # Wikipedia記事:
    {article}

    # 出力形式:
    ### あらすじ
    （ここに要約を記述）

    ### 登場人物
    （ここに要約を記述）

    ### その他
    （ここに要約を記述）
    """

    prompt = PromptTemplate.from_template(template)
    
    # LangChain Expression Language (LCEL) を使ってチェーンを構築
    chain = prompt | llm_pipeline | StrOutputParser()
    
    print("LLMによる要約を生成中...")
    summary = chain.invoke({"article": article_content})
    return summary


if __name__ == "__main__":
    SEARCH_TERM = "クレヨンしんちゃん"
    
    # 1. Wikipediaから記事内容を取得
    content = get_wikipedia_content(SEARCH_TERM)
    
    if content:
        # 2. LLMを初期化
        llm = initialize_llm()

        # 3. テキストをチャンクに分割
        # chunk_sizeは、モデルのトークン限界とプロンプトの長さを考慮して決める
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = [Document(page_content=x) for x in text_splitter.split_text(content)]
        print(f"記事を {len(docs)} 個のチャンクに分割しました。")

        # 4. Map-Reduceチェーンの定義と実行
        # mapプロンプト: 各チャンクをどう要約するかの指示
        map_prompt_template = """
        以下の文章は、あるWikipedia記事の一部です。後で他の部分と結合して最終的な要約を作成します。
        この部分の要点を簡潔にまとめてください。

        ---
        {text}
        ---

        要約:
        """
        
        # combineプロンプト: 個別の要約をどう統合するかの指示
        combine_prompt_template = """
        以下の文章は、あるWikipedia記事の各部分を要約したものです。
        これらの情報全体を元に、指定された3つの観点から内容を要約し、構造化された形式で出力してください。

        # 指示:
        - あらすじ: 物語の基本的な設定やストーリーの概要。
        - 登場人物: 主要なキャラクターやその関係性。
        - その他: メディア展開、社会的影響、制作背景など。

        各セクションは、見出し（例: `### あらすじ`）から始めてください。

        ---
        {text}
        ---

        # 構造化された最終要約:
        """
        
        # LangChainの要約チェーンをロード
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=PromptTemplate.from_template(map_prompt_template),
            combine_prompt=PromptTemplate.from_template(combine_prompt_template),
            verbose=True # 処理の途中経過を表示する
        )
        
        # チェーンの実行
        summary = chain.invoke(docs)
        
        # 結果の表示
        print("\n" + "="*50)
        print("【生成された構造化要約】")
        print("="*50)
        print(summary['output_text'])
# stage2_vl_correct.py
import os
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import torch
from PIL import Image  # 依存はしてるが画像はここでは使わない（将来拡張用）
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ==================================================
# GLOBAL PARAMS (EDIT HERE)
# ==================================================
APP_ROOT = "/app"  # Docker: -v $(PWD):/app

EPISODE_ID = "episode01"

# Stage1 outputs
STAGE1_OUT_ROOT = f"outputs/{EPISODE_ID}"
STAGE1_INDEX = f"{STAGE1_OUT_ROOT}/index.stage1.json"

# Stage0 outputs (script knowledge base)
SCRIPT_PHASE = "script_phase1"
STAGE0_OUT_ROOT = f"outputs/{SCRIPT_PHASE}/{EPISODE_ID}"
LEXICON_PATH = f"{STAGE0_OUT_ROOT}/lexicon_entities.json"
SCRIPT_CHUNKS_JSONL = f"{STAGE0_OUT_ROOT}/script_chunks.jsonl"

# Stage2 outputs
STAGE2_OUT_ROOT = f"outputs_stage2/{EPISODE_ID}"
STAGE2_CUTS_DIR = f"{STAGE2_OUT_ROOT}/cuts"
STAGE2_INDEX = f"{STAGE2_OUT_ROOT}/index.stage2.json"

# Symbol lexicon (制作記号辞書)
SYMBOL_LEXICON_PATH = f"data/storyboard_symbol_lexicon.json"
# ==================================================

# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
CFG = {
    # which columns to merge from stage1
    "use_columns": ["action_memo", "dialogue"],

    # placeholder collapse
    "placeholder_square": "□",
    "placeholder_triangle": "△",

    # script RAG
    "script_top_k": 3,
    "script_min_score": 0.66,  # ルールベース版と同等：高スコアのみ残す

    # symbol glossary injection
    "glossary_max_terms": 512,  # 全部入れると長いので上限（まずは十分大きめ）

    # VL model for correction
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,
    "gen": {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1
    },

    # safety: retry on invalid json
    "vl_retry": 1,
}
# ==================================================


# -------------------------
# Basic helpers
# -------------------------
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
QUOTE_ONLY_RE = re.compile(r"「([^」]+)」")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def norm_spaces(s: str) -> str:
    s = (s or "")
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def collapse_placeholder_lines(text: str, placeholder: str) -> str:
    """
    ルールベース版と同様：
      - '□' だけの行が連続する場合、1つにまとめる
      - 意味のある行はそのまま
    """
    if not text:
        return text
    out = []
    prev = False
    for ln in text.splitlines():
        t = ln.strip()
        is_ph = (t == placeholder)
        if is_ph:
            if not prev:
                out.append(placeholder)
            prev = True
        else:
            out.append(ln)
            prev = False
    return "\n".join(out)

def join_lines_from_stage1_cell(cell: Dict[str, Any]) -> str:
    """
    Stage1 cell: {"raw_text":..., "lines":[...]}
    """
    if not cell:
        return ""
    raw = cell.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    lines = cell.get("lines")
    if isinstance(lines, list):
        return "\n".join([str(x) for x in lines]).strip()
    return ""


# -------------------------
# Script KB loading (Stage0)
# -------------------------
@dataclass
class ScriptChunk:
    chunk_id: str
    paragraph_index: int
    text: str
    kind: str
    scene_id: Optional[str]
    speaker: Optional[str]
    speaker_note: Optional[str]

def load_script_chunks(path_jsonl: str) -> List[ScriptChunk]:
    chunks: List[ScriptChunk] = []
    for obj in iter_jsonl(path_jsonl):
        chunks.append(ScriptChunk(
            chunk_id=str(obj.get("chunk_id", "")),
            paragraph_index=int(obj.get("paragraph_index", -1)),
            text=str(obj.get("text", "")),
            kind=str(obj.get("kind", "")),
            scene_id=obj.get("scene_id"),
            speaker=obj.get("speaker"),
            speaker_note=obj.get("speaker_note"),
        ))
    return chunks

def load_lexicon_characters(path_json: str) -> List[str]:
    obj = read_json(path_json)
    chars = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            name = e.get("canonical")
            if isinstance(name, str) and name.strip():
                chars.append(name.strip())
    chars.sort(key=lambda x: (-len(x), x))
    return chars


# -------------------------
# Symbol glossary loading
# -------------------------
def load_storyboard_symbol_words(path: str) -> List[str]:
    """
    storyboard_symbol_lexicon.json の word をすべて収集。
    意味は使わず、用語羅列のみをプロンプトへ注入する。
    """
    if not path or not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    seen = set()
    if isinstance(data, list):
        for item in data:
            w = item.get("word")
            if isinstance(w, list):
                for s in w:
                    if isinstance(s, str):
                        t = s.strip()
                        if t and t not in seen:
                            seen.add(t)
                            words.append(t)
            elif isinstance(w, str):
                t = w.strip()
                if t and t not in seen:
                    seen.add(t)
                    words.append(t)

    # 長い順（過剰に短い記号が先に来ないように）
    words.sort(key=lambda x: (-len(x), x))
    return words[: CFG["glossary_max_terms"]]


# -------------------------
# RAG (script retrieval) implementation
# -------------------------
def extract_quote_text_for_query(dialogue_raw: str) -> str:
    """
    ルールベース版と同じ思想：
      - 絵コンテにある「」内の台詞が脚本にもある前提
      - 台詞をクエリとして脚本chunksから検索する
    """
    quotes = QUOTE_ONLY_RE.findall(dialogue_raw or "")
    quotes = [q.strip() for q in quotes if q.strip() and q.strip() != CFG["placeholder_square"]]
    quotes = [q.strip() for q in quotes if q.strip() and q.strip() != CFG["placeholder_triangle"]]
    if quotes:
        return "\n".join(quotes)

    # fallback（台詞が取れない場合）
    lines = []
    for ln in (dialogue_raw or "").splitlines():
        t = ln.strip()
        if not t or t == CFG["placeholder_square"]:
            continue
        if not t or t == CFG["placeholder_triangle"]:
            continue
        if t.startswith("S：") or t.startswith("S:"):
            continue
        if re.fullmatch(r"\(.*\)", t):
            continue
        lines.append(t)
    return "\n".join(lines)

def simple_overlap_score(query: str, text: str) -> float:
    """
    RAG検索（簡易版）
    - 文字バイグラムの重なりでスコア付け
    - ライブラリ無しで動く
    - ルールベース版と同じアプローチ（min_scoreで足切り可能）
    """
    q = norm_spaces(query).replace(" ", "")
    t = norm_spaces(text).replace(" ", "")
    if not q or not t:
        return 0.0
    if len(q) < 2 or len(t) < 2:
        return 0.0

    qb = {q[i:i+2] for i in range(len(q)-1)}
    tb = {t[i:i+2] for i in range(len(t)-1)}
    if not qb or not tb:
        return 0.0
    return len(qb & tb) / max(len(qb), 1)

def retrieve_script_chunks(query: str, script_chunks: List[ScriptChunk], top_k: int, min_score: float) -> List[Dict[str, Any]]:
    """
    -------- RAGの中核（コードでの実装説明） --------

    1) query（絵コンテOCRから抽出した台詞など）を作る
       - extract_quote_text_for_query(dialogue_raw) で「」内の台詞を優先

    2) script_chunks.jsonl を段落/台詞単位に分割済みの chunk とみなし、
       各 chunk に対して類似度スコアを計算（simple_overlap_score）

    3) スコア上位 top_k を抽出し、min_score で足切り

    4) 抽出した chunk を "外部知識" として VLモデルの補正プロンプトへ注入

    これが Retrieval-Augmented Generation (RAG) の「Retrieval」部分に相当する。
    """
    scored: List[Tuple[float, ScriptChunk]] = []
    for ch in script_chunks:
        score = simple_overlap_score(query, ch.text)
        if score <= 0:
            continue
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, ch in scored[:top_k]:
        if score < min_score:
            continue
        out.append({
            "chunk_id": ch.chunk_id,
            "score": round(float(score), 4),
            "paragraph_index": ch.paragraph_index,
            "kind": ch.kind,
            "scene_id": ch.scene_id,
            "speaker": ch.speaker,
            "speaker_note": ch.speaker_note,
            "text": ch.text,
        })
    return out


# -------------------------
# VL correction prompt + generation
# -------------------------
def build_vl_correction_prompt(
    *,
    action_raw: str,
    dialogue_raw: str,
    time_raw: str,
    script_evidence: List[Dict[str, Any]],
    symbol_words: List[str],
    lexicon_characters: List[str],
) -> str:
    """
    ここでは「意味」は入れず、用語の羅列のみを注入（あなたの方針）。
    脚本はRAGで取れたchunkだけ注入（コンテキストの過剰入力を避ける）。
    """
    glossary_block = ""
    if symbol_words:
        glossary_block = "=== STORYBOARD TERMS (terms only) ===\n" + "\n".join(symbol_words) + "\n\n"

    # lexiconは名前補正のヒントとして使う（意味は不要）
    # ここも羅列のみ
    char_block = ""
    if lexicon_characters:
        # 長いのが多いので先頭のみ少し（必要なら増やせる）
        char_block = "=== CHARACTER NAMES (canonical, terms only) ===\n" + "\n".join(lexicon_characters[:200]) + "\n\n"

    # RAGで引いた脚本chunkを注入
    script_block = "=== SCRIPT CONTEXT (retrieved chunks) ===\n"
    if not script_evidence:
        script_block += "(none)\n\n"
    else:
        for c in script_evidence:
            # なるべく短く、でも根拠IDは残す
            script_block += f"- speaker={c.get('speaker')} {c.get('text')}\n"
        script_block += "\n"

    # 出力スキーマ：ルールベース版と同等の構造を保ちつつ、補正ログを必須化
    # rawは上書きせず、normだけ出させる（安全）
    prompt = (
        "You are a careful proofreader for storyboard OCR.\n"
        "You will correct OCR text using the provided SCRIPT CONTEXT, CHARACTER NAMES, and STORYBOARD TERMS.\n\n"

        "Hard constraints:\n"
        "- Do NOT invent new content.\n"
        "- Do NOT rewrite stylistically.\n"
        "- Only fix obvious OCR errors.\n"
        "- If uncertain, leave unchanged.\n\n"

        "Replacement constraints (IMPORTANT):\n"
        "- Character name corrections: You may replace a character name ONLY if the corrected name appears in\n"
        "  (A) CHARACTER NAMES or (B) SCRIPT CONTEXT.\n"
        "- Production term/symbol corrections: You may replace a production term ONLY if the corrected term appears in\n"
        "  STORYBOARD TERMS.\n"
        "- Do NOT replace ordinary words using the term list.\n\n"

        "Logging constraints:\n"
        "- If you change any substring, you MUST add an entry to corrections.\n"
        "- Each corrections entry must include evidence_chunk_ids when the SCRIPT CONTEXT supports the change.\n"
        "- If a correction is supported only by the term list, set evidence_chunk_ids to an empty array.\n\n"

        f"{char_block}"
        f"{script_block}"
        f"{glossary_block}"

        "=== INPUT OCR (raw) ===\n"
        f"[action_memo]\n{action_raw}\n\n"
        f"[dialogue]\n{dialogue_raw}\n\n"
        f"[time]\n{time_raw}\n\n"

        "=== OUTPUT (JSON only) ===\n"
        "Return a JSON object with keys:\n"
        "- action_memo_norm: string\n"
        "- dialogue_norm: string\n"
        "- time_norm: string\n"
        "- corrections: array of objects {field, from, to, reason, evidence_chunk_ids}\n"
        "- used_script_chunk_ids: array of chunk_id strings\n"
        "- used_terms: array of terms you used (subset of STORYBOARD TERMS and CHARACTER NAMES)\n\n"

        "Output requirements:\n"
        "- Keep line breaks as much as possible.\n"
        "- Do not add extra keys.\n"
    )
    return prompt


def load_vl_model():
    kwargs = {
        "device_map": CFG["device_map"],
        "torch_dtype": CFG["torch_dtype"],
    }
    if CFG["flash_attn2"]:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(CFG["model_id"], **kwargs)
    processor = AutoProcessor.from_pretrained(CFG["model_id"])
    model.eval()
    return model, processor


@torch.no_grad()
def vl_generate_json(model, processor, prompt: str) -> Dict[str, Any]:
    """
    画像は使わず text-only で補正させる。
    Qwen3-VL はテキストのみでも動く。
    """
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out_ids = model.generate(**inputs, **CFG["gen"])
    trimmed = out_ids[0][len(inputs["input_ids"][0]):]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()

    # JSON抽出（余計な文字があっても最外の{}を拾う）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        raise


# -------------------------
# Stage2 processing
# -------------------------
def list_stage1_cuts(index_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    cuts = index_obj.get("cuts", [])
    out = []
    for c in cuts:
        out_json = c.get("out_json")
        if out_json:
            out.append({"cut": c.get("cut"), "out_json": out_json})
    return out


def stage2_process_one_cut_vl(
    stage1_cut_path: str,
    model,
    processor,
    script_chunks: List[ScriptChunk],
    symbol_words: List[str],
    lexicon_characters: List[str],
) -> Dict[str, Any]:
    obj = read_json(stage1_cut_path)
    cut_num = obj.get("cut")
    span = obj.get("span")
    rows = obj.get("rows", [])

    # ---- Stage1 -> raw merge (same as rule-based stage2) ----
    col_texts_raw: Dict[str, List[str]] = {c: [] for c in CFG["use_columns"]}
    for r in rows:
        cols = r.get("cols", {})
        for col in CFG["use_columns"]:
            cell = cols.get(col, {})
            txt = join_lines_from_stage1_cell(cell)
            if txt:
                col_texts_raw[col].append(txt)

    action_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("action_memo", [])), CFG["placeholder_square"])
    dialogue_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("dialogue", [])), CFG["placeholder_square"])
    time_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("time", [])), CFG["placeholder_square"])

    action_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("action_memo", [])), CFG["placeholder_triangle"])
    dialogue_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("dialogue", [])), CFG["placeholder_triangle"])
    time_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("time", [])), CFG["placeholder_triangle"])

    # ---- RAG retrieval (script) ----
    query = extract_quote_text_for_query(dialogue_raw).strip()
    if not query:
        # 台詞が無い場合はアクションメモから（保険）
        query = action_raw.strip()

    script_evidence = retrieve_script_chunks(
        query=query,
        script_chunks=script_chunks,
        top_k=CFG["script_top_k"],
        min_score=CFG["script_min_score"],
    )

    # ---- VL correction ----
    prompt = build_vl_correction_prompt(
        action_raw=action_raw,
        dialogue_raw=dialogue_raw,
        time_raw=time_raw,
        script_evidence=script_evidence,
        symbol_words=symbol_words,
        lexicon_characters=lexicon_characters,
    )

    # print("prompt:\n", prompt)

    # 生成（JSON）
    last_err = None
    for _ in range(CFG["vl_retry"] + 1):
        try:
            corrected = vl_generate_json(model, processor, prompt)
            last_err = None
            break
        except Exception as e:
            last_err = e
            corrected = None

    if corrected is None:
        # 失敗したらルールベースと同様に raw をそのまま返す（安全）
        corrected = {
            "action_memo_norm": action_raw,
            "dialogue_norm": dialogue_raw,
            "time_norm": time_raw,
            "corrections": [],
            "used_script_chunk_ids": [],
            "used_terms": [],
            "error": str(last_err) if last_err else "unknown",
        }

    # ---- Output ----
    out = {
        "cut": cut_num,
        "span": span,
        "source_stage1": stage1_cut_path,

        # rawは保持
        "ocr_raw": {
            "action_memo": action_raw,
            "dialogue": dialogue_raw,
            "time_raw": time_raw,
            "time_tokens": TIME_RE.findall("\n".join([time_raw, action_raw, dialogue_raw])),
        },

        # normはVL補正結果
        "ocr_norm": {
            "action_memo": corrected.get("action_memo_norm", action_raw),
            "dialogue": corrected.get("dialogue_norm", dialogue_raw),
            "time_norm": corrected.get("time_norm", time_raw),
        },

        # ルールベース版と同じく補正ログを保持（VLが出す）
        "corrections": corrected.get("corrections", []),

        # RAGで注入した根拠（後処理/説明可能性のため保持）
        "rag": {
            "script_query": query,
            "script_evidence": script_evidence,
            "used_script_chunk_ids": corrected.get("used_script_chunk_ids", []),
            "used_terms": corrected.get("used_terms", []),
            "symbol_lexicon_path": SYMBOL_LEXICON_PATH,
        },

        "meta": {
            "created_at": now_iso(),
            "stage2_version": "vl_correction_rag",
            "model_id": CFG["model_id"],
            "gen": CFG["gen"],
        }
    }
    return out


def main():
    # --- sanity checks ---
    for p in [os.path.join(APP_ROOT, STAGE1_INDEX),
              os.path.join(APP_ROOT, LEXICON_PATH),
              os.path.join(APP_ROOT, SCRIPT_CHUNKS_JSONL)]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    ensure_dir(os.path.join(APP_ROOT, STAGE2_OUT_ROOT))
    ensure_dir(os.path.join(APP_ROOT, STAGE2_CUTS_DIR))

    stage1_index = read_json(os.path.join(APP_ROOT, STAGE1_INDEX))
    stage1_cut_refs = list_stage1_cuts(stage1_index)

    lexicon_characters = load_lexicon_characters(os.path.join(APP_ROOT, LEXICON_PATH))
    script_chunks = load_script_chunks(os.path.join(APP_ROOT, SCRIPT_CHUNKS_JSONL))
    symbol_words = load_storyboard_symbol_words(os.path.join(APP_ROOT, SYMBOL_LEXICON_PATH))

    # load VL model once
    model, processor = load_vl_model()

    index_out = {
        "episode_id": EPISODE_ID,
        "source": {
            "stage1_index": STAGE1_INDEX,
            "lexicon": LEXICON_PATH,
            "script_chunks": SCRIPT_CHUNKS_JSONL,
            "symbol_lexicon": SYMBOL_LEXICON_PATH,
        },
        "cuts": [],
        "meta": {
            "created_at": now_iso(),
            "stage2_version": "vl_correction_rag",
            "cfg": CFG,
            "stats": {
                "lexicon_characters": len(lexicon_characters),
                "script_chunks": len(script_chunks),
                "symbol_terms": len(symbol_words),
            }
        }
    }

    for ref in stage1_cut_refs:
        cut_num = ref.get("cut")
        stage1_cut_path = ref["out_json"]
        if not os.path.isabs(stage1_cut_path):
            stage1_cut_path = os.path.join(APP_ROOT, stage1_cut_path)

        if not os.path.isfile(stage1_cut_path):
            index_out["cuts"].append({
                "cut": cut_num,
                "stage1_path": stage1_cut_path,
                "stage2_path": None,
                "status": "missing_stage1_file",
            })
            continue

        stage2_obj = stage2_process_one_cut_vl(
            stage1_cut_path=stage1_cut_path,
            model=model,
            processor=processor,
            script_chunks=script_chunks,
            symbol_words=symbol_words,
            lexicon_characters=lexicon_characters,
        )

        if cut_num is None:
            out_name = f"cut_unknown_{len(index_out['cuts']):06d}.stage2.json"
        else:
            out_name = f"cut{int(cut_num):04d}.stage2.json"
        out_path = os.path.join(APP_ROOT, STAGE2_CUTS_DIR, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stage2_obj, f, ensure_ascii=False, indent=2)

        index_out["cuts"].append({
            "cut": cut_num,
            "stage1_path": stage1_cut_path,
            "stage2_path": out_path,
            "status": "ok",
        })

        print(f"[OK] wrote {out_path}")

    with open(os.path.join(APP_ROOT, STAGE2_INDEX), "w", encoding="utf-8") as f:
        json.dump(index_out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {os.path.join(APP_ROOT, STAGE2_INDEX)}")


if __name__ == "__main__":
    main()

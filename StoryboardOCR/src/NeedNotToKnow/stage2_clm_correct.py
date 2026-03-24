# stage2_clm_correct.py
# Stage2 (Causal LM): OCR結果の整形・補正（画像なし）
#
# 要件（この版で実装）:
# (1) キャラ名補正（辞書照合＋LM判断）            -> 差分ログに "stage=char"
# (2) 制作用語補正（用語辞書RAG＋LM判断）          -> 差分ログに "stage=term"
# (3) セリフ補正（脚本RAG＋LM判断）                -> 差分ログに "stage=dialogue"
# (4) 日本語として正しいか確認して補正（任意）      -> 差分ログに "stage=jp"
#
# さらに:
# - どこが補正されたか分かるように、(from->to)を "diff_span" で記録（最小の差分範囲）
# - (4)あり/なしを比較できるように、concat出力を2種類保存:
#     * cutXXXX.stage2.concat.nojp.json  （(1)-(3)のみ）
#     * cutXXXX.stage2.concat.jp.json    （(1)-(4)）
#   split出力も同様に2種類:
#     * cutXXXX.stage2.split.nojp.json
#     * cutXXXX.stage2.split.jp.json
#
# RAG方針:
# - 距離計算（類似度）は脚本チャンク検索（retrieval）にのみ使用
# - キャラ名/用語/日本語整形は LM に判断させる（辞書/用語/脚本をプロンプト注入）
#
# 実装メモ:
# - LM出力は "修正後テキストのみ"（JSONを要求しない）→ フォーマット崩れを減らす
# - 差分ログはコード側で生成（確実に残る）
#
# モデル: llm-jp/llm-jp-3-8x1.8b-instruct3 （CFGで変更可）

import os
import re
import json
import glob
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================================================
# GLOBAL PARAMS (EDIT HERE)
# ==================================================
APP_ROOT = "/app"
EPISODE_ID = "episode01"

# Stage1 inputs
STAGE1_CUTS_DIR = f"{APP_ROOT}/outputs/{EPISODE_ID}/cuts"

# Stage0 inputs
SCRIPT_PHASE = "script_phase1"
STAGE0_OUT_ROOT = f"{APP_ROOT}/outputs/{SCRIPT_PHASE}/{EPISODE_ID}"
LEXICON_PATH = f"{STAGE0_OUT_ROOT}/lexicon_entities.json"
SCRIPT_CHUNKS_JSONL = f"{STAGE0_OUT_ROOT}/script_chunks.jsonl"

# Symbol lexicon
SYMBOL_LEXICON_PATH = f"{APP_ROOT}/data/storyboard_symbol_lexicon.json"

# Stage2 outputs
STAGE2_OUT_ROOT = f"{APP_ROOT}/outputs_stage2_clm/{EPISODE_ID}"
STAGE2_SPLIT_DIR = f"{STAGE2_OUT_ROOT}/cuts_split"
STAGE2_CONCAT_DIR = f"{STAGE2_OUT_ROOT}/cuts_concat"
STAGE2_INDEX = f"{STAGE2_OUT_ROOT}/index.stage2.json"
# ==================================================


# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
CFG = {
    "cut_select": "all",  # "all" | "range"
    "cut_start": 1,
    "cut_end": 10,

    # Which columns to correct
    "columns": ["action_memo", "dialogue"],

    # Placeholder normalization
    "placeholders": ["□", "△"],

    # Script retrieval
    "script_top_k": 3,
    "script_min_score": 0.60,

    # Term retrieval
    "term_max_k": 40,

    # CLM model (swap later)
    "model_id": "llm-jp/llm-jp-3-8x1.8b-instruct3",
    # "model_id": "llm-jp/llm-jp-3-8x13b-instruct3",

    "torch_dtype": torch.bfloat16,
    "device_map": "auto",

    # Generation (deterministic)
    "gen": {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    },

    # Compare with/without stage (4)
    "enable_japanese_polish": True,     # if True, write both nojpn and jp outputs
    "polish_only_for": ["dialogue", "action_memo"],  # usually time should not be polished

    # Debug
    "print_some": False,
}
# ==================================================


# -------------------------
# Utilities
# -------------------------
CUT_RE = re.compile(r"cut(\d+)\.stage1\.json$")
QUOTE_RE = re.compile(r"「([^」]+)」")

OUT_BEGIN = "<<<OUT>>>"
OUT_END = "<<<END>>>"


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


def cut_in_range(c: Optional[int]) -> bool:
    if CFG["cut_select"] == "all":
        return True
    if c is None:
        return False
    return CFG["cut_start"] <= c <= CFG["cut_end"]


def normalize_text_basic(s: str) -> str:
    s = s or ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collapse_placeholder_lines(text: str) -> str:
    """
    連続する □/△ 行を1つにまとめる
    """
    if not text:
        return text
    ph = set(CFG["placeholders"])
    out = []
    prev_ph = False
    for ln in text.splitlines():
        t = ln.strip()
        is_ph = (t in ph)
        if is_ph:
            if not prev_ph:
                out.append(t)
            prev_ph = True
        else:
            out.append(ln)
            prev_ph = False
    return "\n".join(out)


def extract_quotes(text: str) -> List[str]:
    return [q.strip() for q in QUOTE_RE.findall(text or "") if q.strip()]


def compute_diff_span(a: str, b: str) -> Dict[str, Any]:
    """
    Return minimal differing span between a and b.
    """
    if a == b:
        return {"changed": False}

    i = 0
    min_len = min(len(a), len(b))
    while i < min_len and a[i] == b[i]:
        i += 1

    j = 0
    a_rem = a[i:]
    b_rem = b[i:]
    min_rem = min(len(a_rem), len(b_rem))
    while j < min_rem and a_rem[-(j + 1)] == b_rem[-(j + 1)]:
        j += 1

    a_start = i
    a_end = len(a) - j
    b_start = i
    b_end = len(b) - j

    return {
        "changed": True,
        "a_span": [a_start, max(a_start, a_end)],
        "b_span": [b_start, max(b_start, b_end)],
        "from_sub": a[a_start:a_end],
        "to_sub": b[b_start:b_end],
    }


def json_safe(obj: Any) -> Any:
    """
    Convert non-JSON-serializable types to serializable equivalents.
    - torch.dtype -> string
    - torch.device -> string
    - set -> list
    - recursively applies to dict/list/tuple
    """
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    # basic types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # fallback to string
    return str(obj)


# -------------------------
# Load lexicons
# -------------------------
def load_character_lexicon(path: str) -> List[str]:
    obj = read_json(path)
    names = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            n = e.get("canonical")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())

    out = []
    seen = set()
    for n in sorted(names, key=lambda x: (-len(x), x)):
        if n not in seen:
            seen.add(n)
            out.append(n)
        if "・" in n:
            short = n.split("・")[0]
            if short and short not in seen:
                seen.add(short)
                out.append(short)
    return out


def load_symbol_lexicon(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Return term -> {category, meaning}
    """
    data = read_json(path)
    term2 = {}
    if isinstance(data, list):
        for item in data:
            cat = item.get("category", "unknown")
            meaning = item.get("意味", "")
            words = item.get("word", [])
            if not isinstance(words, list):
                continue
            for w in words:
                if isinstance(w, str) and w.strip():
                    term2[w.strip()] = {"category": cat, "meaning": meaning}
    return term2


# -------------------------
# Script chunks + RAG retrieval (distance used ONLY here)
# -------------------------
@dataclass
class ScriptChunk:
    chunk_id: str
    text: str
    kind: str
    scene_id: Optional[str]
    speaker: Optional[str]


def load_script_chunks(path: str) -> List[ScriptChunk]:
    out = []
    for obj in iter_jsonl(path):
        out.append(
            ScriptChunk(
                chunk_id=str(obj.get("chunk_id", "")),
                text=str(obj.get("text", "")),
                kind=str(obj.get("kind", "")),
                scene_id=obj.get("scene_id"),
                speaker=obj.get("speaker"),
            )
        )
    return out


def bigrams(s: str) -> set:
    s = normalize_text_basic(s).replace(" ", "")
    if len(s) < 2:
        return set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def simple_overlap_score(query: str, text: str) -> float:
    qb = bigrams(query)
    tb = bigrams(text)
    if not qb or not tb:
        return 0.0
    return len(qb & tb) / max(len(qb), 1)


def retrieve_script_candidates(query: str, script_chunks: List[ScriptChunk]) -> List[Dict[str, Any]]:
    scored = []
    for ch in script_chunks:
        score = simple_overlap_score(query, ch.text)
        if score <= 0:
            continue
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, ch in scored[: CFG["script_top_k"]]:
        if score < CFG["script_min_score"]:
            continue
        out.append({
            "chunk_id": ch.chunk_id,
            "score": round(float(score), 4),
            "kind": ch.kind,
            "scene_id": ch.scene_id,
            "speaker": ch.speaker,
            "text": ch.text,
        })
    return out


# -------------------------
# Term retrieval (no distance; simple match)
# -------------------------
def retrieve_terms_in_text(text: str, term2info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    found = []
    seen = set()
    terms_sorted = sorted(term2info.keys(), key=lambda x: (-len(x), x))

    for term in terms_sorted:
        if term in text and term not in seen:
            seen.add(term)
            info = term2info[term]
            found.append({"term": term, "category": info.get("category"), "meaning": info.get("meaning", "")})
            if len(found) >= CFG["term_max_k"]:
                break
    return found


# -------------------------
# Causal LM wrapper (llm-jp)
# -------------------------
SYSTEM_JA = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"


def load_clm():
    tok = AutoTokenizer.from_pretrained(CFG["model_id"])
    # Ensure pad_token exists
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        device_map=CFG["device_map"],
        torch_dtype=CFG["torch_dtype"],
    )
    model.eval()
    return tok, model


@torch.no_grad()
def clm_generate(tokenizer, model, system: str, user: str) -> str:
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    batch = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    batch = {k: v.to(model.device) for k, v in batch.items()}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    out_ids = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pad_token_id=pad_id,
        **CFG["gen"],
    )[0]

    # ★重要：入力長以降だけをデコード
    in_len = batch["input_ids"].shape[1]
    gen_only = out_ids[in_len:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()


def extract_last_assistant_text(decoded: str) -> str:
    m = re.search(r"(?:assistant|###\s*応答|###\s*Response)\s*[:：]?\s*", decoded, flags=re.IGNORECASE)
    if m:
        return decoded[m.end():].strip()
    return decoded.strip()


# -------------------------
# Prompts (type-specific)
# -------------------------
def prompt_character_correction(text: str, character_terms: List[str], script_candidates: List[Dict[str, Any]]) -> str:
    chars_block = "\n".join(character_terms[:120]) if character_terms else "(none)"
    script_block = "\n".join([f"[{c['chunk_id']}] speaker={c.get('speaker')} text={c.get('text')}" for c in script_candidates]) or "(none)"

    return (
        "あなたはOCRの補正係です。\n"
        "タスク：登場人物名の誤認識だけを最小限で補正してください。\n"
        "制約：登場人物名以外（一般文・記号・英字・用語・句読点・改行）は変更しない。\n"
        "不確かな場合は変更しない。\n\n"
        f"出力は必ず次のタグの間に「補正後テキストのみ」を出してください。\n"
        f"{OUT_BEGIN}\n(ここに補正後テキスト)\n{OUT_END}\n"
        "タグ以外は一切出力しない。見出しや説明も禁止。\n\n"
        "登場人物名候補（用語の羅列）:\n"
        f"{chars_block}\n\n"
        "脚本候補（参考、変更根拠にしてよい）:\n"
        f"{script_block}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
    )


def prompt_term_correction(text: str, terms_with_meaning: List[Dict[str, Any]]) -> str:
    block = "\n".join([f"- {t['term']} ({t.get('category','')}): {t.get('meaning','')}" for t in terms_with_meaning]) or "(none)"

    return (
        "あなたはOCRの補正係です。\n"
        "タスク：アニメ制作の記号・用語の誤認識だけを最小限で補正してください。\n"
        "制約：一般文や台詞の内容は変更しない。勝手に言い換えない。\n"
        "制約：見出し（例：補正後テキスト:）や説明文を出力しない。\n"
        "不確かな場合は変更しない。\n\n"
        f"出力は必ず次のタグの間に「補正後テキストのみ」を出してください。\n"
        f"{OUT_BEGIN}\n(ここに補正後テキスト)\n{OUT_END}\n\n"
        "用語候補（このカットで検出されたもの。意味つき）:\n"
        f"{block}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
    )


def prompt_dialogue_correction(text: str, script_candidates: List[Dict[str, Any]]) -> str:
    script_block = "\n".join([f"[{c['chunk_id']}] speaker={c.get('speaker')} text={c.get('text')}" for c in script_candidates]) or "(none)"

    return (
        "あなたはOCRの補正係です。\n"
        "タスク：台詞（「」の中身）と話者名の誤認識を最小限で補正してください。\n"
        "方針：脚本候補に同じ台詞がある場合は、その表記を優先して合わせる。\n"
        "制約：脚本に無い内容を追加しない。台詞の意味を変えない。文体を変えない。\n"
        "制約：SEなど効果音タグや記号は維持する。\n"
        "制約：見出し（例：補正後テキスト:）や説明文は禁止。\n"
        "不確かな場合は変更しない。\n\n"
        f"出力は必ず次のタグの間に「補正後テキストのみ」を出してください。\n"
        f"{OUT_BEGIN}\n(ここに補正後テキスト)\n{OUT_END}\n\n"
        "脚本候補（RAG）:\n"
        f"{script_block}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
    )


def prompt_japanese_polish(text: str) -> str:
    return (
        "あなたは日本語校正係です。\n"
        "タスク：明らかな誤字脱字・変換ミスだけを最小限で補正してください。\n"
        "最重要：意味の言い換え・要約・説明の追加は禁止。\n"
        "最重要：英字/記号/制作用語（PAN, wipe, SE, O.L, 矢印など）は一切変更しない。\n"
        "最重要：行数・改行は維持する（削除や追加をしない）。\n"
        "不確かな場合は変更しない。\n\n"
        f"出力は必ず次のタグの間に「補正後テキストのみ」を出してください。\n"
        f"{OUT_BEGIN}\n(ここに補正後テキスト)\n{OUT_END}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
    )


# -------------------------
# Correction helpers (stage-aware logs)
# -------------------------

def extract_tagged_output(text: str) -> str:
    s = text.find(OUT_BEGIN)
    e = text.find(OUT_END)
    if s != -1 and e != -1 and e > s:
        return text[s + len(OUT_BEGIN): e].strip()
    # タグが無い場合はそのまま（保険）
    return text.strip()


def corrected_or_same(tokenizer, model, user_prompt: str, raw_text: str) -> str:
    decoded = clm_generate(tokenizer, model, SYSTEM_JA, user_prompt)
    ans = extract_tagged_output(decoded)
    return ans if ans.strip() else raw_text


def build_correction_log(stage: str, field: str, before: str, after: str, reason: str, evidence_ids: List[str]) -> List[Dict[str, Any]]:
    if before == after:
        return []
    return [{
        "stage": stage,
        "field": field,
        "from": before,
        "to": after,
        "reason": reason,
        "evidence_chunk_ids": evidence_ids,
        "diff_span": compute_diff_span(before, after),
    }]


def concat_rows_text(rows: List[Dict[str, Any]], col: str, use_key: str) -> str:
    parts = []
    for r in rows:
        c = (r.get("cols", {}) or {}).get(col, {}) or {}
        t = c.get(use_key, "")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
    return collapse_placeholder_lines("\n".join(parts))


# -------------------------
# Stage2 pipeline
# -------------------------
def list_stage1_files() -> List[str]:
    return sorted(glob.glob(os.path.join(STAGE1_CUTS_DIR, "cut*.stage1.json")))


def main():
    ensure_dir(STAGE2_OUT_ROOT)
    ensure_dir(STAGE2_SPLIT_DIR)
    ensure_dir(STAGE2_CONCAT_DIR)

    character_terms = load_character_lexicon(LEXICON_PATH)
    term2info = load_symbol_lexicon(SYMBOL_LEXICON_PATH)
    script_chunks = load_script_chunks(SCRIPT_CHUNKS_JSONL)

    tokenizer, model = load_clm()

    stage1_files = list_stage1_files()

    index_out: Dict[str, Any] = {
        "episode_id": EPISODE_ID,
        "source": {
            "stage1_cuts_dir": STAGE1_CUTS_DIR,
            "lexicon": LEXICON_PATH,
            "script_chunks": SCRIPT_CHUNKS_JSONL,
            "symbol_lexicon": SYMBOL_LEXICON_PATH,
            "clm_model_id": CFG["model_id"],
        },
        "cuts": [],
        "meta": {
            "created_at": now_iso(),
            "stage2_version": "clm_correction_split_concat_with_optional_jp",
            "cfg": json_safe(CFG),  # ★ dtypeなどをJSON-safe化
        }
    }

    for s1_path in tqdm(stage1_files, desc="Stage2 CLM: cuts"):
        m = CUT_RE.search(os.path.basename(s1_path))
        cut_num = int(m.group(1)) if m else None
        if not cut_in_range(cut_num):
            continue

        stage1 = read_json(s1_path)
        rows = stage1.get("rows", [])

        # --- Build cut-level query for script retrieval (distance used ONLY here) ---
        dialogue_parts = []
        for row_obj in rows:
            cols = row_obj.get("cols", {}) or {}
            dg = (cols.get("dialogue", {}) or {}).get("raw_text", "") or ""
            if isinstance(dg, str) and dg.strip():
                dialogue_parts.append(dg)
        dialogue_all = "\n".join(dialogue_parts)

        quotes = extract_quotes(dialogue_all)
        query = "\n".join(quotes).strip() if quotes else normalize_text_basic(dialogue_all)

        script_candidates = retrieve_script_candidates(query, script_chunks)

        # --- Term RAG from symbol lexicon (no distance) ---
        cut_text_for_terms_parts = []
        for row_obj in rows:
            cols = row_obj.get("cols", {}) or {}
            am = (cols.get("action_memo", {}) or {}).get("raw_text", "") or ""
            dg = (cols.get("dialogue", {}) or {}).get("raw_text", "") or ""
            if isinstance(am, str) and am.strip():
                cut_text_for_terms_parts.append(am)
            if isinstance(dg, str) and dg.strip():
                cut_text_for_terms_parts.append(dg)
        cut_text_for_terms = "\n".join(cut_text_for_terms_parts)

        terms_with_meaning = retrieve_terms_in_text(cut_text_for_terms, term2info)

        # Build split base
        base_split = {
            "cut": stage1.get("cut"),
            "cut_str": stage1.get("cut_str"),
            "span": stage1.get("span"),
            "source_stage1": s1_path,
            "rows": [],
            "rag": {
                "script_query": query,
                "script_candidates": script_candidates[:CFG["script_top_k"]],
                "used_script_chunk_ids": [c["chunk_id"] for c in script_candidates[:CFG["script_top_k"]]],
                "terms_with_meaning": terms_with_meaning[:CFG["term_max_k"]],
            },
            "meta": {
                "created_at": now_iso(),
                "model_id": CFG["model_id"],
                "gen": json_safe(CFG["gen"]),
            }
        }

        # Two variants: nojp and jp
        split_nojp = {**base_split, "stage2_variant": "nojp", "corrections": []}
        split_jp = {**base_split, "stage2_variant": "jp", "corrections": []}

        # Process each row
        for row_obj in rows:
            page = row_obj.get("page")
            row = row_obj.get("row")
            cols = row_obj.get("cols", {}) or {}

            row_out_nojp = {"page": page, "row": row, "page_image": row_obj.get("page_image"), "cols": {}}
            row_out_jp = {"page": page, "row": row, "page_image": row_obj.get("page_image"), "cols": {}}

            for col_name, cell in cols.items():
                cell = cell or {}
                raw = str(cell.get("raw_text", "") or "")
                raw_norm = collapse_placeholder_lines(raw)

                corrected_nojp = raw_norm

                # Apply (1)-(3)
                if col_name in CFG["columns"]:
                    # (2) term correction
                    if col_name in ("action_memo", "dialogue"):
                        p = prompt_term_correction(corrected_nojp, terms_with_meaning)
                        out_term = corrected_or_same(tokenizer, model, p, corrected_nojp)
                        split_nojp["corrections"].extend(build_correction_log(
                            stage="term",
                            field=col_name,
                            before=corrected_nojp,
                            after=out_term,
                            reason="term_correction_with_symbol_rag",
                            evidence_ids=[],
                        ))
                        corrected_nojp = out_term

                    # (1) character correction
                    if col_name in ("action_memo", "dialogue"):
                        p = prompt_character_correction(corrected_nojp, character_terms, script_candidates[:CFG["script_top_k"]])
                        out_char = corrected_or_same(tokenizer, model, p, corrected_nojp)
                        split_nojp["corrections"].extend(build_correction_log(
                            stage="char",
                            field=col_name,
                            before=corrected_nojp,
                            after=out_char,
                            reason="character_name_correction_with_lexicon_and_script",
                            evidence_ids=[c["chunk_id"] for c in script_candidates[:CFG["script_top_k"]]],
                        ))
                        corrected_nojp = out_char

                    # (3) dialogue correction
                    if col_name == "dialogue":
                        p = prompt_dialogue_correction(corrected_nojp, script_candidates[:CFG["script_top_k"]])
                        out_dlg = corrected_or_same(tokenizer, model, p, corrected_nojp)
                        split_nojp["corrections"].extend(build_correction_log(
                            stage="dialogue",
                            field=col_name,
                            before=corrected_nojp,
                            after=out_dlg,
                            reason="dialogue_alignment_with_script_rag",
                            evidence_ids=[c["chunk_id"] for c in script_candidates[:CFG["script_top_k"]]],
                        ))
                        corrected_nojp = out_dlg

                # JP variant starts from nojp
                corrected_jp = corrected_nojp
                if CFG["enable_japanese_polish"] and (col_name in CFG["polish_only_for"]):
                    p = prompt_japanese_polish(corrected_jp)
                    out_jp = corrected_or_same(tokenizer, model, p, corrected_jp)
                    split_jp["corrections"].extend(build_correction_log(
                        stage="jp",
                        field=col_name,
                        before=corrected_jp,
                        after=out_jp,
                        reason="japanese_polish_minimal",
                        evidence_ids=[],
                    ))
                    corrected_jp = out_jp

                row_out_nojp["cols"][col_name] = {**cell, "corrected_text": corrected_nojp}
                row_out_jp["cols"][col_name] = {**cell, "corrected_text": corrected_jp}

            split_nojp["rows"].append(row_out_nojp)
            split_jp["rows"].append(row_out_jp)

        # Concat builders
        def build_concat(split_obj: Dict[str, Any], variant: str) -> Dict[str, Any]:
            rows2 = split_obj["rows"]
            return {
                "cut": split_obj["cut"],
                "cut_str": split_obj["cut_str"],
                "span": split_obj["span"],
                "source_stage1": split_obj["source_stage1"],
                "source_stage2_split": None,
                "stage2_variant": variant,
                "ocr_raw": {
                    "action_memo": concat_rows_text(rows2, "action_memo", use_key="raw_text"),
                    "dialogue": concat_rows_text(rows2, "dialogue", use_key="raw_text"),
                    "time_raw": concat_rows_text(rows2, "time", use_key="raw_text"),
                },
                "ocr_norm": {
                    "action_memo": concat_rows_text(rows2, "action_memo", use_key="corrected_text"),
                    "dialogue": concat_rows_text(rows2, "dialogue", use_key="corrected_text"),
                    "time_norm": concat_rows_text(rows2, "time", use_key="corrected_text"),
                },
                "rag": split_obj["rag"],
                "corrections": split_obj["corrections"],
                "meta": {
                    "created_at": now_iso(),
                    "model_id": CFG["model_id"],
                    "gen": json_safe(CFG["gen"]),
                    "stage2_type": "concat",
                }
            }

        concat_nojp = build_concat(split_nojp, "nojp")
        concat_jp = build_concat(split_jp, "jp")

        # Write files
        cut_name = f"cut{int(split_nojp['cut']):04d}" if split_nojp["cut"] is not None else "cut_unknown"

        split_nojp_path = os.path.join(STAGE2_SPLIT_DIR, f"{cut_name}.stage2.split.nojp.json")
        concat_nojp_path = os.path.join(STAGE2_CONCAT_DIR, f"{cut_name}.stage2.concat.nojp.json")

        split_jp_path = os.path.join(STAGE2_SPLIT_DIR, f"{cut_name}.stage2.split.jp.json")
        concat_jp_path = os.path.join(STAGE2_CONCAT_DIR, f"{cut_name}.stage2.concat.jp.json")

        concat_nojp["source_stage2_split"] = split_nojp_path
        concat_jp["source_stage2_split"] = split_jp_path

        with open(split_nojp_path, "w", encoding="utf-8") as f:
            json.dump(split_nojp, f, ensure_ascii=False, indent=2)
        with open(concat_nojp_path, "w", encoding="utf-8") as f:
            json.dump(concat_nojp, f, ensure_ascii=False, indent=2)

        if CFG["enable_japanese_polish"]:
            with open(split_jp_path, "w", encoding="utf-8") as f:
                json.dump(split_jp, f, ensure_ascii=False, indent=2)
            with open(concat_jp_path, "w", encoding="utf-8") as f:
                json.dump(concat_jp, f, ensure_ascii=False, indent=2)

        index_out["cuts"].append({
            "cut": split_nojp["cut"],
            "stage1": s1_path,
            "stage2_split_nojp": split_nojp_path,
            "stage2_concat_nojp": concat_nojp_path,
            "stage2_split_jp": split_jp_path if CFG["enable_japanese_polish"] else None,
            "stage2_concat_jp": concat_jp_path if CFG["enable_japanese_polish"] else None,
        })

    with open(STAGE2_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {STAGE2_INDEX}")


if __name__ == "__main__":
    main()

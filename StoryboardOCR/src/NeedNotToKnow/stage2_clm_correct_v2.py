# stage2_clm_correct_v2.py
# ------------------------------------------------------------
# Stage2 (Causal LM): OCR結果の整形・補正（画像なし）
#
# Fixes vs previous:
# - NO shallow-copy bug (nojp/jp rows never mix)
# - Robust tag extraction (even if <<<END>>> missing)
# - Term prompt does NOT include meanings (prevents explanation outputs)
# - Skip dialogue correction if script_candidates is empty
# - Output guard: reject suspicious outputs (RAG labels, explanations, too-large edits)
#
# Model: llm-jp/llm-jp-3-8x1.8b-instruct3 (CFG)
# ------------------------------------------------------------

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

    # Which columns to correct by LM stages (term/char/dialogue/jp)
    "columns": ["action_memo", "dialogue"],

    # Placeholder normalization
    "placeholders": ["□", "△"],

    # Script retrieval
    "script_top_k": 3,
    "script_min_score": 0.60,

    # Term retrieval
    "term_max_k": 40,

    # CLM model
    "model_id": "llm-jp/llm-jp-3-8x1.8b-instruct3",
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
    "enable_japanese_polish": True,
    "polish_only_for": ["dialogue", "action_memo"],

    # Output guard (very important)
    # - if output contains these patterns, reject and keep original
    "reject_regex": [
        r"\bRAG\b",
        r"脚本候補",
        r"用語候補",
        r"補正後",
        r"説明",
        r"意味[:：]",
        r"<<<OUT>>>",
        r"<<<END>>>",
    ],
    # If new long english tokens appear (>=5 chars) and not in allowlist, reject
    "reject_new_english_longword": True,
    "allowed_english_tokens": [
        "PAN", "FIX", "TILT", "ZOOM", "WIPE", "CUT", "IN", "OUT", "OFF", "ON",
        "SE", "BGM", "MA", "O.L", "OL", "CONT", "CONTINUED"
    ],
    # Stage-wise maximum length ratio and maximum diff span
    "guard": {
        "term":    {"max_len_ratio": 1.15, "max_diff_span": 30},
        "char":    {"max_len_ratio": 1.15, "max_diff_span": 30},
        "dialogue": {"max_len_ratio": 1.25, "max_diff_span": 120},
        "jp":      {"max_len_ratio": 1.25, "max_diff_span": 120},
    },

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

SYSTEM_JA = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"


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
    If multiple distant edits exist, the span becomes large (good for guard).
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
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
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
    (meaning is stored in code, but NOT injected into term correction prompt)
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
# Script chunks + retrieval (distance used ONLY here)
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
# Causal LM wrapper
# -------------------------
def load_clm():
    tok = AutoTokenizer.from_pretrained(CFG["model_id"])
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

    in_len = batch["input_ids"].shape[1]
    gen_only = out_ids[in_len:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()


# -------------------------
# Output extraction + sanitization + guard
# -------------------------
def extract_tagged_output_flexible(text: str) -> str:
    """
    Robust extraction:
    - If both tags exist -> extract between
    - If only OUT_BEGIN exists -> take after it (until OUT_END if later, else end-of-text)
    - If no tags -> return stripped text
    """
    if not text:
        return ""

    s = text.find(OUT_BEGIN)
    e = text.find(OUT_END)

    if s != -1 and e != -1 and e > s:
        return text[s + len(OUT_BEGIN): e].strip()

    if s != -1:
        # take after OUT_BEGIN
        tail = text[s + len(OUT_BEGIN):]
        e2 = tail.find(OUT_END)
        if e2 != -1:
            tail = tail[:e2]
        return tail.strip()

    if e != -1:
        # if only END exists, take before it (rare)
        return text[:e].strip()

    return text.strip()


def strip_markers(s: str) -> str:
    """
    Remove any leftover markers or code fences.
    """
    if not s:
        return s
    s = s.replace(OUT_BEGIN, "").replace(OUT_END, "")
    # remove fenced blocks if any
    s = re.sub(r"^```.*?\n", "", s, flags=re.DOTALL)
    s = s.replace("```", "")
    return s.strip()


def contains_reject_pattern(s: str) -> Optional[str]:
    for pat in CFG.get("reject_regex", []):
        if re.search(pat, s, flags=re.IGNORECASE):
            return pat
    return None


def english_tokens(s: str) -> set:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\.\-]{1,}", s or "")
    return set(toks)


def guard_output(stage: str, before: str, after: str) -> Tuple[bool, str]:
    """
    Return (ok, reason_if_rejected).
    """
    if after is None:
        return False, "after_none"
    a = after.strip()
    if not a:
        return False, "after_empty"

    # reject obvious patterns
    pat = contains_reject_pattern(a)
    if pat:
        return False, f"reject_regex:{pat}"

    # length ratio guard
    g = (CFG.get("guard", {}) or {}).get(stage, {})
    max_len_ratio = float(g.get("max_len_ratio", 10.0))
    if len(before) > 0 and len(a) > int(len(before) * max_len_ratio):
        return False, f"too_long:{len(a)}/{len(before)} ratio>{max_len_ratio}"

    # diff span guard
    diff = compute_diff_span(before, a)
    if diff.get("changed"):
        span_len = max(len(diff.get("from_sub", "")), len(diff.get("to_sub", "")))
        max_span = int(g.get("max_diff_span", 10**9))
        if span_len > max_span:
            return False, f"diff_span_too_large:{span_len}>{max_span}"

    # reject new long english tokens
    if CFG.get("reject_new_english_longword", True):
        allow = set([t.upper() for t in CFG.get("allowed_english_tokens", [])])
        bset = set([t.upper() for t in english_tokens(before)])
        aset = set([t.upper() for t in english_tokens(a)])
        new = aset - bset
        for t in new:
            if t in allow:
                continue
            if len(t) >= 5:
                return False, f"new_english_token:{t}"

    return True, ""


def corrected_or_same(
    tokenizer,
    model,
    *,
    stage: str,
    user_prompt: str,
    raw_text: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate -> extract -> sanitize -> guard.
    If rejected, return raw_text.
    """
    decoded = clm_generate(tokenizer, model, SYSTEM_JA, user_prompt)
    extracted = extract_tagged_output_flexible(decoded)
    cleaned = strip_markers(extracted)

    ok, reason = guard_output(stage, raw_text, cleaned)
    if not ok:
        return raw_text, {
            "stage": stage,
            "accepted": False,
            "reject_reason": reason,
            "model_raw": decoded[:500],
        }

    return cleaned, {
        "stage": stage,
        "accepted": True,
        "reject_reason": "",
    }


# -------------------------
# Prompts (type-specific)
# -------------------------
EXAMPLE_FMT = (
    "出力例:\n"
    f"{OUT_BEGIN}\n"
    "（入力テキストをできるだけ維持しつつ、必要最小限のみ補正した結果）\n"
    f"{OUT_END}\n"
)

def prompt_character_correction(text: str, character_terms: List[str], script_candidates: List[Dict[str, Any]]) -> str:
    chars_block = "\n".join(character_terms[:120]) if character_terms else "(none)"
    script_block = "\n".join([f"[{c['chunk_id']}] speaker={c.get('speaker')} text={c.get('text')}" for c in script_candidates]) or "(none)"

    return (
        "あなたはOCRの補正係です。\n"
        "タスク：登場人物名の誤認識だけを必要最小限で補正してください。\n"
        "最重要制約：登場人物名以外（一般文・記号・英字・用語・句読点・改行）は変更しない。\n"
        "最重要制約：入力に無い語を追加しない。\n"
        "不確かな場合は変更しない。\n\n"
        "出力フォーマットは必ずタグで囲む。\n"
        f"{EXAMPLE_FMT}\n"
        "タグ以外は一切出力しない。見出しや説明は禁止。\n\n"
        "登場人物名候補（用語の羅列）:\n"
        f"{chars_block}\n\n"
        "脚本候補（参考。空なら無視してよい）:\n"
        f"{script_block}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
        "\n必ず次の形式で出力:\n"
        f"{OUT_BEGIN}\n"
        "(ここに補正後テキストのみ)\n"
        f"{OUT_END}\n"
    )


def prompt_term_correction(text: str, terms_with_info: List[Dict[str, Any]]) -> str:
    # IMPORTANT: do NOT include meaning in the prompt (prevents explanation)
    block = "\n".join([f"- {t['term']} ({t.get('category','')})" for t in terms_with_info]) or "(none)"

    return (
        "あなたはOCRの補正係です。\n"
        "タスク：アニメ制作の記号・用語の誤認識だけを必要最小限で補正してください。\n"
        "最重要制約：入力に無い用語を新しく追加しない。\n"
        "最重要制約：用語の意味説明や注釈を出力しない。\n"
        "最重要制約：一般文や台詞の内容は変更しない。勝手に言い換えない。\n"
        "最重要制約：改行を維持する（行の追加・削除をしない）。\n"
        "不確かな場合は変更しない。\n\n"
        "出力フォーマットは必ずタグで囲む。\n"
        f"{EXAMPLE_FMT}\n"
        "タグ以外は一切出力しない。見出しや説明は禁止。\n\n"
        "用語候補（このテキスト内で検出された用語の羅列。出力に説明を書かない）:\n"
        f"{block}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
        "\n必ず次の形式で出力:\n"
        f"{OUT_BEGIN}\n"
        "(ここに補正後テキストのみ)\n"
        f"{OUT_END}\n"
    )


def prompt_dialogue_correction(text: str, script_candidates: List[Dict[str, Any]]) -> str:
    script_block = "\n".join([f"[{c['chunk_id']}] speaker={c.get('speaker')} text={c.get('text')}" for c in script_candidates]) or "(none)"

    return (
        "あなたはOCRの補正係です。\n"
        "タスク：台詞（「」の中身）と話者名の誤認識を必要最小限で補正してください。\n"
        "方針：脚本候補に同じ台詞がある場合のみ、その表記に合わせる。\n"
        "最重要制約：脚本候補が (none) の場合は一切変更しない。\n"
        "最重要制約：脚本に無い内容を追加しない。台詞の意味を変えない。文体を変えない。\n"
        "最重要制約：記号やタグ（SE/MA/BGM/ガヤ/off/on）や改行を維持する。\n"
        "最重要制約：『RAG』などのラベル文字列を出力に含めない。\n"
        "不確かな場合は変更しない。\n\n"
        "出力フォーマットは必ずタグで囲む。\n"
        f"{EXAMPLE_FMT}\n"
        "タグ以外は一切出力しない。見出しや説明は禁止。\n\n"
        "脚本候補（RAG）:\n"
        f"{script_block}\n\n"
        "入力テキスト:\n"
        f"{text}\n"
        "\n必ず次の形式で出力:\n"
        f"{OUT_BEGIN}\n"
        "(ここに補正後テキストのみ)\n"
        f"{OUT_END}\n"
    )


def prompt_japanese_polish(text: str) -> str:
    return (
        "あなたは日本語校正係です。\n"
        "タスク：明らかな誤字脱字・変換ミスだけを必要最小限で補正してください。\n"
        "最重要制約：意味の言い換え・要約・説明の追加は禁止。\n"
        "最重要制約：英字/記号/制作用語（PAN, FIX, SE, O.L, 矢印など）は一切変更しない。\n"
        "最重要制約：行数・改行は維持する（削除や追加をしない）。\n"
        "不確かな場合は変更しない。\n\n"
        "出力フォーマットは必ずタグで囲む。\n"
        f"{EXAMPLE_FMT}\n"
        "タグ以外は一切出力しない。見出しや説明は禁止。\n\n"
        "入力テキスト:\n"
        f"{text}\n"
        "\n必ず次の形式で出力:\n"
        f"{OUT_BEGIN}\n"
        "(ここに補正後テキストのみ)\n"
        f"{OUT_END}\n"
    )


# -------------------------
# Correction logs
# -------------------------
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


def make_split_base(
    *,
    stage1_obj: Dict[str, Any],
    s1_path: str,
    query: str,
    script_candidates: List[Dict[str, Any]],
    terms_with_meaning: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "cut": stage1_obj.get("cut"),
        "cut_str": stage1_obj.get("cut_str"),
        "span": stage1_obj.get("span"),
        "source_stage1": s1_path,
        "rag": {
            "script_query": query,
            "script_candidates": script_candidates[:CFG["script_top_k"]],
            "used_script_chunk_ids": [c["chunk_id"] for c in script_candidates[:CFG["script_top_k"]]],
            "terms_with_meaning": terms_with_meaning[:CFG["term_max_k"]],  # stored for later, not necessarily injected
        },
        "meta": {
            "created_at": now_iso(),
            "model_id": CFG["model_id"],
            "gen": json_safe(CFG["gen"]),
        }
    }


def make_split_variant(base: Dict[str, Any], variant: str) -> Dict[str, Any]:
    # IMPORTANT: create new lists per variant (no shared references)
    out = {
        "cut": base["cut"],
        "cut_str": base["cut_str"],
        "span": base["span"],
        "source_stage1": base["source_stage1"],
        "rag": base["rag"],   # read-only
        "meta": base["meta"], # read-only
        "stage2_variant": variant,
        "rows": [],
        "corrections": [],
        "debug": [],
    }
    return out


def build_concat(split_obj: Dict[str, Any], variant: str, source_stage2_split: str) -> Dict[str, Any]:
    rows2 = split_obj["rows"]
    return {
        "cut": split_obj["cut"],
        "cut_str": split_obj["cut_str"],
        "span": split_obj["span"],
        "source_stage1": split_obj["source_stage1"],
        "source_stage2_split": source_stage2_split,
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
            "stage2_version": "clm_correction_split_concat_with_optional_jp_v2",
            "cfg": json_safe(CFG),
        }
    }

    for s1_path in tqdm(stage1_files, desc="Stage2 CLM v2: cuts"):
        m = CUT_RE.search(os.path.basename(s1_path))
        cut_num = int(m.group(1)) if m else None
        if not cut_in_range(cut_num):
            continue

        stage1 = read_json(s1_path)
        rows = stage1.get("rows", [])

        # --- Build cut-level query for script retrieval ---
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

        # --- Term retrieval from symbol lexicon ---
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

        base = make_split_base(
            stage1_obj=stage1,
            s1_path=s1_path,
            query=query,
            script_candidates=script_candidates,
            terms_with_meaning=terms_with_meaning,
        )

        split_nojp = make_split_variant(base, "nojp")
        split_jp = make_split_variant(base, "jp")

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
                debug_notes: List[str] = []

                # Apply (2)->(1)->(3) only for configured columns
                if col_name in CFG["columns"]:
                    # (2) term correction
                    p = prompt_term_correction(corrected_nojp, terms_with_meaning)
                    out_term, dbg = corrected_or_same(
                        tokenizer, model,
                        stage="term",
                        user_prompt=p,
                        raw_text=corrected_nojp
                    )
                    if not dbg["accepted"]:
                        debug_notes.append(f"reject(term):{dbg['reject_reason']}")
                    split_nojp["corrections"].extend(build_correction_log(
                        stage="term",
                        field=col_name,
                        before=corrected_nojp,
                        after=out_term,
                        reason="term_correction_with_symbol_lexicon_terms_only",
                        evidence_ids=[],
                    ))
                    corrected_nojp = out_term

                    # (1) character correction
                    p = prompt_character_correction(corrected_nojp, character_terms, script_candidates[:CFG["script_top_k"]])
                    out_char, dbg = corrected_or_same(
                        tokenizer, model,
                        stage="char",
                        user_prompt=p,
                        raw_text=corrected_nojp
                    )
                    if not dbg["accepted"]:
                        debug_notes.append(f"reject(char):{dbg['reject_reason']}")
                    split_nojp["corrections"].extend(build_correction_log(
                        stage="char",
                        field=col_name,
                        before=corrected_nojp,
                        after=out_char,
                        reason="character_name_correction_with_lexicon_and_script",
                        evidence_ids=[c["chunk_id"] for c in script_candidates[:CFG["script_top_k"]]],
                    ))
                    corrected_nojp = out_char

                    # (3) dialogue correction (ONLY if script candidates exist)
                    if col_name == "dialogue" and len(script_candidates) > 0:
                        p = prompt_dialogue_correction(corrected_nojp, script_candidates[:CFG["script_top_k"]])
                        out_dlg, dbg = corrected_or_same(
                            tokenizer, model,
                            stage="dialogue",
                            user_prompt=p,
                            raw_text=corrected_nojp
                        )
                        if not dbg["accepted"]:
                            debug_notes.append(f"reject(dialogue):{dbg['reject_reason']}")
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
                    out_jp, dbg = corrected_or_same(
                        tokenizer, model,
                        stage="jp",
                        user_prompt=p,
                        raw_text=corrected_jp
                    )
                    if not dbg["accepted"]:
                        debug_notes.append(f"reject(jp):{dbg['reject_reason']}")
                    split_jp["corrections"].extend(build_correction_log(
                        stage="jp",
                        field=col_name,
                        before=corrected_jp,
                        after=out_jp,
                        reason="japanese_polish_minimal",
                        evidence_ids=[],
                    ))
                    corrected_jp = out_jp

                # store outputs (tags already stripped)
                row_out_nojp["cols"][col_name] = {
                    **cell,
                    "corrected_text": corrected_nojp,
                    "stage2_notes": debug_notes,
                }
                row_out_jp["cols"][col_name] = {
                    **cell,
                    "corrected_text": corrected_jp,
                    "stage2_notes": debug_notes,
                }

            split_nojp["rows"].append(row_out_nojp)
            split_jp["rows"].append(row_out_jp)

        # Write files
        cut_name = f"cut{int(split_nojp['cut']):04d}" if split_nojp["cut"] is not None else "cut_unknown"

        split_nojp_path = os.path.join(STAGE2_SPLIT_DIR, f"{cut_name}.stage2.split.nojp.json")
        concat_nojp_path = os.path.join(STAGE2_CONCAT_DIR, f"{cut_name}.stage2.concat.nojp.json")

        split_jp_path = os.path.join(STAGE2_SPLIT_DIR, f"{cut_name}.stage2.split.jp.json")
        concat_jp_path = os.path.join(STAGE2_CONCAT_DIR, f"{cut_name}.stage2.concat.jp.json")

        with open(split_nojp_path, "w", encoding="utf-8") as f:
            json.dump(split_nojp, f, ensure_ascii=False, indent=2)
        concat_nojp = build_concat(split_nojp, "nojp", split_nojp_path)
        with open(concat_nojp_path, "w", encoding="utf-8") as f:
            json.dump(concat_nojp, f, ensure_ascii=False, indent=2)

        if CFG["enable_japanese_polish"]:
            with open(split_jp_path, "w", encoding="utf-8") as f:
                json.dump(split_jp, f, ensure_ascii=False, indent=2)
            concat_jp = build_concat(split_jp, "jp", split_jp_path)
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

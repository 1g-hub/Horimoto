# stage2_clm_correct_singleprompt.py
# ------------------------------------------------------------
# Stage2 (Single-Prompt / Causal LM):
# - 1つのプロンプトテンプレートで、action_memo / dialogue を同時に最小限補正する
# - 画像なし（text-only）
# - Script RAG（簡易バイグラム重なり）で脚本chunkをヒント注入
# - Symbol lexicon / Character lexicon を「用語の羅列」として注入（意味説明は禁止）
# - 出力は JSON（行配列）で返させ、コード側で join して改行を維持
# - 強いガードで「過補正」を弾き、危険な場合は原文維持
# - outputs_stage2_clm/<episode>/cuts_split と cuts_concat を生成（nojp / jp 両対応）
# ------------------------------------------------------------

import os
import re
import json
import glob
import ast
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
STAGE1_CUTS_DIR = f"{APP_ROOT}/outputs_LoRA_pic_act_dia/{EPISODE_ID}/cuts"

# Stage0 inputs
SCRIPT_PHASE = "script_phase1"
STAGE0_OUT_ROOT = f"{APP_ROOT}/outputs/{SCRIPT_PHASE}/{EPISODE_ID}"
LEXICON_PATH = f"{STAGE0_OUT_ROOT}/lexicon_entities.json"
SCRIPT_CHUNKS_JSONL = f"{STAGE0_OUT_ROOT}/script_chunks.jsonl"

# Symbol lexicon
SYMBOL_LEXICON_PATH = f"{APP_ROOT}/data/storyboard_symbol_lexicon.json"

# Stage2 outputs
STAGE2_OUT_ROOT = f"{APP_ROOT}/outputs_stage2_clm_LoRA_pic_act_dia/{EPISODE_ID}"
STAGE2_SPLIT_DIR = f"{STAGE2_OUT_ROOT}/cuts_split"
STAGE2_CONCAT_DIR = f"{STAGE2_OUT_ROOT}/cuts_concat"
STAGE2_INDEX = f"{STAGE2_OUT_ROOT}/index.stage2.json"
# ==================================================


# ==================================================
# CONFIG (EDIT HERE)
#   ※ユーザー提示のCFGを尊重しつつ、足りない項目を安全に追加
# ==================================================
CFG = {
    "cut_select": "all",  # "all" | "range"
    "cut_start": 1,
    "cut_end": 10,

    # Which columns to correct (we still carry others through)
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

    # Compare with/without jp polishing (we run 2 variants if enabled)
    "enable_japanese_polish": True,
    "polish_only_for": ["dialogue", "action_memo"],

    # Output guard
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
    "reject_new_english_longword": True,
    "allowed_english_tokens": [
        "PAN", "FIX", "TILT", "ZOOM", "WIPE", "CUT", "IN", "OUT", "OFF", "ON",
        "SE", "BGM", "MA", "O.L", "OL", "CONT", "CONTINUED"
    ],
    "guard": {
        "term":     {"max_len_ratio": 1.15, "max_diff_span": 30},
        "char":     {"max_len_ratio": 1.15, "max_diff_span": 30},
        "dialogue": {"max_len_ratio": 1.25, "max_diff_span": 120},
        "jp":       {"max_len_ratio": 1.25, "max_diff_span": 120},
    },

    "print_some": False,

    # ---- added (safe defaults) ----
    "all_columns": ["cut", "picture", "action_memo", "dialogue", "time"],
    "max_char_names_in_prompt": 200,
    "max_terms_in_prompt": 80,           # promptに入れる用語（検出済み）上限
    "max_script_chars_per_chunk": 140,   # script chunkテキスト切り詰め
    "single_prompt_retries": 2,
    "force_time_unchanged": True,        # timeはモデル出力を無視して原文維持
    "enforce_same_line_count": True,     # 行数が変わる補正を拒否
}
# ==================================================


# -------------------------
# Regex / constants
# -------------------------
CUT_RE = re.compile(r"cut(\d+)\.stage1\.json$")
QUOTE_RE = re.compile(r"「([^」]+)」")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")

OUT_BEGIN = "<<<OUT>>>"
OUT_END = "<<<END>>>"

SYSTEM_JA = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"


# -------------------------
# Basic helpers
# -------------------------
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

def cut_in_range(cut_num: Optional[int]) -> bool:
    if CFG["cut_select"] == "all":
        return True
    if cut_num is None:
        return False
    return int(CFG["cut_start"]) <= int(cut_num) <= int(CFG["cut_end"])

def norm_spaces(s: str) -> str:
    s = (s or "")
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def collapse_placeholder_lines_multi(text: str, placeholders: List[str]) -> str:
    """
    連続する placeholder 行を 1つにまとめる（placeholderごとに処理）
    """
    if not text:
        return text
    out = text
    for ph in placeholders:
        out2 = []
        prev_ph = False
        for ln in out.splitlines():
            t = ln.strip()
            is_ph = (t == ph)
            if is_ph:
                if not prev_ph:
                    out2.append(ph)
                prev_ph = True
            else:
                out2.append(ln)
                prev_ph = False
        out = "\n".join(out2)
    return out

def join_cell_text(cell: Dict[str, Any]) -> str:
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
# Diff span (guard / logging)
# -------------------------
def compute_diff_span(a: str, b: str) -> Dict[str, Any]:
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


# -------------------------
# Stage0: script chunks + lexicon
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
    out: List[ScriptChunk] = []
    for obj in iter_jsonl(path_jsonl):
        out.append(ScriptChunk(
            chunk_id=str(obj.get("chunk_id", "")),
            paragraph_index=int(obj.get("paragraph_index", -1)),
            text=str(obj.get("text", "")),
            kind=str(obj.get("kind", "")),
            scene_id=obj.get("scene_id"),
            speaker=obj.get("speaker"),
            speaker_note=obj.get("speaker_note"),
        ))
    return out

def load_lexicon_characters(path_json: str) -> List[str]:
    if not os.path.isfile(path_json):
        return []
    obj = read_json(path_json)
    chars = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            name = e.get("canonical")
            if isinstance(name, str) and name.strip():
                chars.append(name.strip())
    # longer first
    chars = sorted(set(chars), key=lambda x: (-len(x), x))
    return chars


# -------------------------
# Symbol lexicon
# -------------------------
def load_symbol_lexicon(path: str) -> Dict[str, Dict[str, Any]]:
    """
    term -> {category, meaning}
    meaningは保存用（promptには入れない）
    """
    if not os.path.isfile(path):
        return {}
    data = read_json(path)
    term2: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            cat = item.get("category", "unknown")
            meaning = item.get("意味", "")
            words = item.get("word", [])
            if not isinstance(words, list):
                continue
            for w in words:
                if isinstance(w, str) and w.strip():
                    term2[w.strip()] = {"category": cat, "meaning": meaning if isinstance(meaning, str) else ""}
    return term2

def retrieve_terms_in_text(text: str, term2info: Dict[str, Dict[str, Any]], max_k: int) -> List[Dict[str, Any]]:
    """
    “文字列として出現した用語”のみ拾う（安全・過補正防止）
    """
    found: List[Dict[str, Any]] = []
    seen = set()
    terms_sorted = sorted(term2info.keys(), key=lambda x: (-len(x), x))

    for term in terms_sorted:
        if term in text and term not in seen:
            seen.add(term)
            info = term2info[term]
            found.append({"term": term, "category": info.get("category"), "meaning": info.get("meaning", "")})
            if len(found) >= max_k:
                break
    return found


# -------------------------
# RAG: script retrieval (bigrams overlap)
# -------------------------
def extract_query_from_dialogue(dialogue_raw: str) -> str:
    """
    「」内を優先してクエリ化。無ければ通常行。
    """
    dialogue_raw = dialogue_raw or ""
    quotes = [q.strip() for q in QUOTE_RE.findall(dialogue_raw) if q.strip()]
    quotes = [q for q in quotes if q not in CFG["placeholders"]]
    if quotes:
        return "\n".join(quotes).strip()

    # fallback: placeholderや空行を除いて繋ぐ
    lines = []
    for ln in dialogue_raw.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t in CFG["placeholders"]:
            continue
        lines.append(t)
    return "\n".join(lines).strip()

def bigram_set(s: str) -> set:
    s = norm_spaces(s).replace(" ", "")
    if len(s) < 2:
        return set()
    return {s[i:i+2] for i in range(len(s)-1)}

def simple_overlap_score(query: str, text: str) -> float:
    qb = bigram_set(query)
    tb = bigram_set(text)
    if not qb or not tb:
        return 0.0
    return len(qb & tb) / max(len(qb), 1)

def retrieve_script_chunks(query: str, script_chunks: List[ScriptChunk], top_k: int, min_score: float) -> List[Dict[str, Any]]:
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
# CLM loading + generation
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
    chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
# Robust extraction + parse
# -------------------------
def extract_tagged_output_flexible(text: str) -> str:
    if not text:
        return ""
    s = text.find(OUT_BEGIN)
    e = text.find(OUT_END)

    if s != -1 and e != -1 and e > s:
        return text[s + len(OUT_BEGIN): e].strip()
    if s != -1:
        tail = text[s + len(OUT_BEGIN):]
        e2 = tail.find(OUT_END)
        if e2 != -1:
            tail = tail[:e2]
        return tail.strip()
    if e != -1:
        return text[:e].strip()
    return text.strip()

def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def extract_brace_block(s: str) -> str:
    s = s or ""
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j+1]
    return s.strip()

def parse_json_relaxed(text: str) -> Dict[str, Any]:
    """
    JSON strict -> brace extraction -> trailing comma fix -> ast literal_eval fallback
    """
    t = strip_code_fences(text)
    t = extract_brace_block(t)
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    t = re.sub(r",\s*}", "}", t)
    t = re.sub(r",\s*]", "]", t)

    try:
        return json.loads(t)
    except Exception:
        pass

    # python-ish dict fallback
    t2 = re.sub(r"\bnull\b", "None", t, flags=re.IGNORECASE)
    t2 = re.sub(r"\btrue\b", "True", t2, flags=re.IGNORECASE)
    t2 = re.sub(r"\bfalse\b", "False", t2, flags=re.IGNORECASE)
    obj = ast.literal_eval(t2)
    if isinstance(obj, dict):
        return obj
    raise json.JSONDecodeError("parse_json_relaxed failed", text, 0)


# -------------------------
# Guard
# -------------------------
def contains_reject_pattern(s: str) -> Optional[str]:
    for pat in CFG.get("reject_regex", []):
        if re.search(pat, s or "", flags=re.IGNORECASE):
            return pat
    return None

def english_tokens(s: str) -> set:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\.\-]{1,}", s or "")
    return set(toks)

def guard_text(
    *,
    field: str,
    before: str,
    after: str,
    variant_stage: str,
    enforce_same_lines: bool,
) -> Tuple[bool, str]:
    """
    variant_stage: "nojp" or "jp" （guardテーブル参照用）
    """
    if after is None:
        return False, "after_none"
    a = after.strip("\n")
    if a == "":
        # 空にするのは危険（原文維持が安全）
        if before.strip() == "":
            return True, ""
        return False, "after_empty"

    # reject patterns
    pat = contains_reject_pattern(a)
    if pat:
        return False, f"reject_regex:{pat}"

    # same line count
    if enforce_same_lines:
        b_lines = before.splitlines()
        a_lines = a.splitlines()
        if len(b_lines) != len(a_lines):
            return False, f"line_count_changed:{len(b_lines)}->{len(a_lines)}"

        # placeholder lines must be unchanged
        ph = set(CFG["placeholders"])
        for bl, al in zip(b_lines, a_lines):
            if bl.strip() in ph and al.strip() != bl.strip():
                return False, "placeholder_line_changed"

    # length ratio guard (fieldごとに適用)
    # action_memo は "term" ガード相当、dialogue は "dialogue"/"jp" を使う
    if field == "action_memo":
        g = (CFG.get("guard", {}) or {}).get("term", {})
    else:
        g = (CFG.get("guard", {}) or {}).get("jp" if variant_stage == "jp" else "dialogue", {})

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

    # new english long word guard
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


# -------------------------
# Single prompt builder (row-level)
# -------------------------
def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + "…"

def build_single_prompt(
    *,
    variant: str,  # "nojp" or "jp"
    cut_num: int,
    page: int,
    row: int,
    action_lines: List[str],
    dialogue_lines: List[str],
    time_lines: List[str],
    script_candidates: List[Dict[str, Any]],
    terms_with_info: List[Dict[str, Any]],
    character_names: List[str],
) -> str:
    """
    出力は行配列のJSON（改行問題を避ける）
    """
    char_block = ""
    if character_names:
        char_block = "=== CHARACTER NAMES (canonical, terms only) ===\n" + "\n".join(character_names[: int(CFG["max_char_names_in_prompt"])]) + "\n\n"

    script_block = "=== SCRIPT CONTEXT (retrieved chunks) ===\n"
    if not script_candidates:
        script_block += "(none)\n\n"
    else:
        for c in script_candidates[: int(CFG["script_top_k"])]:
            cid = str(c.get("chunk_id", ""))
            spk = str(c.get("speaker", "") or "")
            tx = _truncate(str(c.get("text", "") or ""), int(CFG["max_script_chars_per_chunk"]))
            script_block += f"- [{cid}] speaker={spk} text={tx}\n"
        script_block += "\n"

    # IMPORTANT: meaningは入れない（説明誘発を避ける）
    terms_only = [t["term"] for t in terms_with_info if isinstance(t.get("term"), str)]
    terms_only = terms_only[: int(CFG["max_terms_in_prompt"])]
    term_block = ""
    if terms_only:
        term_block = "=== STORYBOARD TERMS (terms only) ===\n" + "\n".join(terms_only) + "\n\n"

    input_obj = {
        "meta": {"cut": cut_num, "page": page, "row": row, "variant": variant},
        "action_memo_lines": action_lines,
        "dialogue_lines": dialogue_lines,
        "time_lines": time_lines,
    }

    jp_clause = ""
    if variant == "jp":
        jp_clause = (
            "追加ルール（jp variant）:\n"
            "- ひらがな/漢字の明らかな誤字脱字だけを最小限で補正してよい。\n"
            "- ただし制作用語・英字・記号（PAN/FIX/SE/矢印/括弧/コロン等）は一切変更しない。\n"
            "- 文体の言い換え・説明の追加は禁止。\n"
        )

    prompt = (
        "あなたはアニメ絵コンテ内テキスト補正係。\n"
        "目的：テキストの誤りのみを与えられた情報から必要最小限の編集で補正して。\n\n"
        "最重要制約（絶対）:\n"
        "- 入力に無い語を追加しない（説明・注釈・意味の追記は禁止）。\n"
        "- 文体の言い換え・要約は禁止。\n"
        "- 改行を維持する：各フィールドの行数を絶対に変えない。\n"
        "- 不確かな場合は変更しない。\n"
        "- 『RAG』などのラベル文字列を出力に含めない。\n\n"
        "許可される補正（厳格）:\n"
        "- 登場人物名：名前が CHARACTER NAMES もしくは SCRIPT CONTEXT に存在する場合のみ置換して。\n"
        "- 制作用語：用語が STORYBOARD TERMS に存在する場合のみ置換して。\n"
        "- 台詞（dialogue）：SCRIPT CONTEXT にほぼ同一台詞が存在する場合のみ表記を合わせて。\n"
        "  SCRIPT CONTEXT が (none) の場合、dialogue は一切変更しない。\n\n"
        f"{jp_clause}\n"
        f"{char_block}"
        f"{script_block}"
        f"{term_block}"
        "=== INPUT (JSON) ===\n"
        f"{json.dumps(input_obj, ensure_ascii=False)}\n\n"
        "=== OUTPUT (JSON only) ===\n"
        "次のキーだけを持つ JSON を、必ずタグの間に出力してください。\n"
        "- action_memo_lines: array<string>  （入力と同じ行数）\n"
        "- dialogue_lines: array<string>     （入力と同じ行数）\n"
        "- time_lines: array<string>         （入力と同じ行数。基本は変更しない）\n"
        "- used_script_chunk_ids: array<string>\n"
        "- used_terms: array<string>\n\n"
        "出力フォーマット（厳守）:\n"
        f"{OUT_BEGIN}\n"
        "{\"action_memo_lines\": [...], \"dialogue_lines\": [...], \"time_lines\": [...], \"used_script_chunk_ids\": [...], \"used_terms\": [...]} \n"
        f"{OUT_END}\n"
        "タグ以外は一切出力しない。\n"
    )
    return prompt


# -------------------------
# Stage1 listing
# -------------------------
def list_stage1_cut_files() -> List[str]:
    return sorted(glob.glob(os.path.join(STAGE1_CUTS_DIR, "cut*.stage1.json")))

def concat_rows_text(rows: List[Dict[str, Any]], col: str, use_key: str) -> str:
    parts = []
    for r in rows:
        c = (r.get("cols", {}) or {}).get(col, {}) or {}
        t = c.get(use_key, "")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
    return collapse_placeholder_lines_multi("\n".join(parts), CFG["placeholders"])


# -------------------------
# Main per-cut processing
# -------------------------
def process_cut_singleprompt(
    *,
    stage1_obj: Dict[str, Any],
    stage1_path: str,
    tokenizer,
    model,
    script_chunks: List[ScriptChunk],
    term2info: Dict[str, Dict[str, Any]],
    character_names: List[str],
    variant: str,  # "nojp" or "jp"
) -> Dict[str, Any]:
    cut_num = int(stage1_obj.get("cut") or 0)
    cut_str = stage1_obj.get("cut_str")
    span = stage1_obj.get("span")
    rows_in = stage1_obj.get("rows", []) or []

    # row order is important -> sort defensively
    def _row_key(ro: Dict[str, Any]) -> Tuple[int, int]:
        try:
            return (int(ro.get("page", 10**9)), int(ro.get("row", 10**9)))
        except Exception:
            return (10**9, 10**9)

    rows_in = sorted(rows_in, key=_row_key)

    # ---- cut-level RAG retrieval context ----
    # build dialogue_all for query
    dialogue_all_parts = []
    text_for_terms_parts = []
    for ro in rows_in:
        cols = ro.get("cols", {}) or {}
        dg = join_cell_text((cols.get("dialogue") or {}))
        am = join_cell_text((cols.get("action_memo") or {}))
        if dg.strip():
            dialogue_all_parts.append(dg.strip())
        if am.strip():
            text_for_terms_parts.append(am.strip())
        if dg.strip():
            text_for_terms_parts.append(dg.strip())

    dialogue_all = collapse_placeholder_lines_multi("\n".join(dialogue_all_parts), CFG["placeholders"])
    query = extract_query_from_dialogue(dialogue_all)
    if not query:
        query = collapse_placeholder_lines_multi("\n".join(text_for_terms_parts), CFG["placeholders"]).strip()

    script_candidates = retrieve_script_chunks(
        query=query,
        script_chunks=script_chunks,
        top_k=int(CFG["script_top_k"]),
        min_score=float(CFG["script_min_score"]),
    )

    terms_with_meaning = retrieve_terms_in_text(
        text=collapse_placeholder_lines_multi("\n".join(text_for_terms_parts), CFG["placeholders"]),
        term2info=term2info,
        max_k=int(CFG["term_max_k"]),
    )

    # ---- split output skeleton ----
    split_obj: Dict[str, Any] = {
        "cut": cut_num,
        "cut_str": cut_str,
        "span": span,
        "source_stage1": stage1_path,
        "stage2_variant": variant,
        "rows": [],
        "rag": {
            "script_query": query,
            "script_candidates": script_candidates[: int(CFG["script_top_k"])],
            "used_script_chunk_ids": [c["chunk_id"] for c in script_candidates[: int(CFG["script_top_k"])]],
            "terms_with_meaning": terms_with_meaning[: int(CFG["term_max_k"])],
        },
        "corrections": [],
        "meta": {
            "created_at": now_iso(),
            "stage2_version": "clm_singleprompt_rowwise",
            "model_id": CFG["model_id"],
            "gen": CFG["gen"],
        }
    }

    # ---- per-row single prompt ----
    for ro in rows_in:
        page = int(ro.get("page", -1) or -1)
        row = int(ro.get("row", -1) or -1)
        cols = ro.get("cols", {}) or {}

        row_out = {
            "page": page,
            "row": row,
            "page_image": ro.get("page_image"),
            "cols": {},
        }

        # gather raw lines
        action_raw = join_cell_text(cols.get("action_memo") or {})
        dialogue_raw = join_cell_text(cols.get("dialogue") or {})
        time_raw = join_cell_text(cols.get("time") or {})

        action_raw = collapse_placeholder_lines_multi(action_raw, CFG["placeholders"])
        dialogue_raw = collapse_placeholder_lines_multi(dialogue_raw, CFG["placeholders"])
        time_raw = collapse_placeholder_lines_multi(time_raw, CFG["placeholders"])

        action_lines = action_raw.splitlines() if action_raw != "" else []
        dialogue_lines = dialogue_raw.splitlines() if dialogue_raw != "" else []
        time_lines = time_raw.splitlines() if time_raw != "" else []

        # If script_candidates empty -> enforce "dialogue unchanged" by instruction + guard
        prompt = build_single_prompt(
            variant=variant,
            cut_num=cut_num,
            page=page,
            row=row,
            action_lines=action_lines,
            dialogue_lines=dialogue_lines,
            time_lines=time_lines,
            script_candidates=script_candidates[: int(CFG["script_top_k"])],
            terms_with_info=terms_with_meaning[: int(CFG["term_max_k"])],
            character_names=character_names,
        )

        decoded = ""
        parsed: Optional[Dict[str, Any]] = None
        last_err: Optional[str] = None

        for _ in range(int(CFG["single_prompt_retries"]) + 1):
            decoded = clm_generate(tokenizer, model, SYSTEM_JA, prompt)
            extracted = extract_tagged_output_flexible(decoded)
            try:
                parsed = parse_json_relaxed(extracted)
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                parsed = None

        # fallback: keep original
        out_action_lines = list(action_lines)
        out_dialogue_lines = list(dialogue_lines)
        out_time_lines = list(time_lines)

        debug_notes: List[str] = []

        if parsed is None:
            debug_notes.append(f"parse_failed:{last_err}")
        else:
            # get lines arrays
            a2 = parsed.get("action_memo_lines")
            d2 = parsed.get("dialogue_lines")
            t2 = parsed.get("time_lines")
            if isinstance(a2, list) and all(isinstance(x, str) for x in a2):
                out_action_lines = a2
            else:
                debug_notes.append("bad_action_memo_lines_type")
            if isinstance(d2, list) and all(isinstance(x, str) for x in d2):
                out_dialogue_lines = d2
            else:
                debug_notes.append("bad_dialogue_lines_type")
            if isinstance(t2, list) and all(isinstance(x, str) for x in t2):
                out_time_lines = t2
            else:
                debug_notes.append("bad_time_lines_type")

        # join + guard
        before_action = "\n".join(action_lines)
        before_dialogue = "\n".join(dialogue_lines)
        before_time = "\n".join(time_lines)

        after_action = "\n".join(out_action_lines)
        after_dialogue = "\n".join(out_dialogue_lines)
        after_time = "\n".join(out_time_lines)

        # force: time unchanged (safer)
        if CFG.get("force_time_unchanged", True):
            after_time = before_time

        # if no script candidates -> force dialogue unchanged (safer)
        if len(script_candidates) == 0:
            after_dialogue = before_dialogue

        # guards
        ok, reason = guard_text(
            field="action_memo",
            before=before_action,
            after=after_action,
            variant_stage=variant,
            enforce_same_lines=bool(CFG.get("enforce_same_line_count", True)),
        )
        if not ok:
            debug_notes.append(f"reject(action_memo):{reason}")
            after_action = before_action

        ok, reason = guard_text(
            field="dialogue",
            before=before_dialogue,
            after=after_dialogue,
            variant_stage=variant,
            enforce_same_lines=bool(CFG.get("enforce_same_line_count", True)),
        )
        if not ok:
            debug_notes.append(f"reject(dialogue):{reason}")
            after_dialogue = before_dialogue

        # (time is unchanged by force)

        # store per-cell corrected_text (carry-through other cols)
        for col_name in CFG["all_columns"]:
            cell = (cols.get(col_name) or {})
            raw_text = join_cell_text(cell)
            raw_text = collapse_placeholder_lines_multi(raw_text, CFG["placeholders"])

            if col_name == "action_memo":
                corrected_text = after_action
            elif col_name == "dialogue":
                corrected_text = after_dialogue
            elif col_name == "time":
                corrected_text = before_time
            else:
                corrected_text = raw_text

            row_out["cols"][col_name] = {
                **cell,
                "corrected_text": corrected_text,
                "stage2_notes": list(debug_notes),
            }

        # correction logs (code-side, reliable)
        if before_action != after_action:
            split_obj["corrections"].append({
                "stage": variant,
                "field": "action_memo",
                "from": before_action,
                "to": after_action,
                "reason": "single_prompt_clm_minimal_edit",
                "evidence_chunk_ids": [],  # term/char等の根拠は断定しない
                "diff_span": compute_diff_span(before_action, after_action),
            })

        if before_dialogue != after_dialogue:
            split_obj["corrections"].append({
                "stage": variant,
                "field": "dialogue",
                "from": before_dialogue,
                "to": after_dialogue,
                "reason": "single_prompt_clm_minimal_edit",
                "evidence_chunk_ids": [c["chunk_id"] for c in script_candidates[: int(CFG["script_top_k"])]],
                "diff_span": compute_diff_span(before_dialogue, after_dialogue),
            })

        split_obj["rows"].append(row_out)

    return split_obj


def build_concat_from_split(split_obj: Dict[str, Any], source_stage2_split: str) -> Dict[str, Any]:
    rows2 = split_obj.get("rows", []) or []
    variant = split_obj.get("stage2_variant", "nojp")

    return {
        "cut": split_obj.get("cut"),
        "cut_str": split_obj.get("cut_str"),
        "span": split_obj.get("span"),
        "source_stage1": split_obj.get("source_stage1"),
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
        "rag": split_obj.get("rag", {}),
        "corrections": split_obj.get("corrections", []),
        "meta": {
            "created_at": now_iso(),
            "model_id": CFG["model_id"],
            "gen": CFG["gen"],
            "stage2_type": "concat",
            "stage2_version": split_obj.get("meta", {}).get("stage2_version"),
        }
    }


# -------------------------
# Entry
# -------------------------
def main():
    # sanity checks
    for p in [STAGE1_CUTS_DIR, LEXICON_PATH, SCRIPT_CHUNKS_JSONL, SYMBOL_LEXICON_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path not found: {p}")

    ensure_dir(STAGE2_OUT_ROOT)
    ensure_dir(STAGE2_SPLIT_DIR)
    ensure_dir(STAGE2_CONCAT_DIR)

    # load KBs
    character_names = load_lexicon_characters(LEXICON_PATH)
    script_chunks = load_script_chunks(SCRIPT_CHUNKS_JSONL)
    term2info = load_symbol_lexicon(SYMBOL_LEXICON_PATH)

    # load model once
    tokenizer, model = load_clm()

    # list stage1 cuts
    stage1_files = list_stage1_cut_files()
    if not stage1_files:
        raise FileNotFoundError(f"No stage1 cut files found: {STAGE1_CUTS_DIR}")

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
            "stage2_version": "clm_singleprompt_rowwise",
            "cfg": CFG,
            "stats": {
                "character_names": len(character_names),
                "script_chunks": len(script_chunks),
                "symbol_terms": len(term2info),
            }
        }
    }

    for s1_path in tqdm(stage1_files, desc="Stage2 single-prompt: cuts"):
        m = CUT_RE.search(os.path.basename(s1_path))
        cut_num = int(m.group(1)) if m else None
        if not cut_in_range(cut_num):
            continue

        stage1_obj = read_json(s1_path)
        cut_val = stage1_obj.get("cut")
        if cut_val is None:
            # unknown cut -> skip or still process; here we still process but name becomes unknown
            cut_val = cut_num

        # run variants
        variants = ["nojp"]
        if CFG.get("enable_japanese_polish", False):
            variants.append("jp")

        cut_name = f"cut{int(cut_val):04d}" if cut_val is not None else "cut_unknown"

        cut_entry: Dict[str, Any] = {
            "cut": cut_val,
            "stage1": s1_path,
            "stage2_split_nojp": None,
            "stage2_concat_nojp": None,
            "stage2_split_jp": None,
            "stage2_concat_jp": None,
        }

        for variant in variants:
            split_obj = process_cut_singleprompt(
                stage1_obj=stage1_obj,
                stage1_path=s1_path,
                tokenizer=tokenizer,
                model=model,
                script_chunks=script_chunks,
                term2info=term2info,
                character_names=character_names,
                variant=variant,
            )

            split_path = os.path.join(STAGE2_SPLIT_DIR, f"{cut_name}.stage2.split.{variant}.json")
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split_obj, f, ensure_ascii=False, indent=2)

            concat_obj = build_concat_from_split(split_obj, source_stage2_split=split_path)
            concat_path = os.path.join(STAGE2_CONCAT_DIR, f"{cut_name}.stage2.concat.{variant}.json")
            with open(concat_path, "w", encoding="utf-8") as f:
                json.dump(concat_obj, f, ensure_ascii=False, indent=2)

            if variant == "nojp":
                cut_entry["stage2_split_nojp"] = split_path
                cut_entry["stage2_concat_nojp"] = concat_path
            else:
                cut_entry["stage2_split_jp"] = split_path
                cut_entry["stage2_concat_jp"] = concat_path

            print(f"[OK] wrote split={split_path}")
            print(f"[OK] wrote concat={concat_path}")

        index_out["cuts"].append(cut_entry)

    with open(STAGE2_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_out, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {STAGE2_INDEX}")


if __name__ == "__main__":
    main()

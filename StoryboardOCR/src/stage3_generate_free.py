# stage3_generate_free_from_stage1.py
#
# Stage3 (FREE dataset / Stage1 OCR):
# - Input: cutXXXX.stage1.json (each cut has multiple rows, each row has its own picture crop)
# - For EACH ROW:
#     - Input image = picture crop only (row_obj["cols"]["picture"]["image"])
#     - Prompt uses:
#         (A) current row text block (raw + corrected if exists; stage1 has raw only)
#         (B) whole-cut ordered text block (raw + corrected if exists)
#         (C) production terms with meanings (from storyboard_symbol_lexicon.json)
#     - Generate:
#         structured.json (STRICT JSON) + description.txt (Japanese)
#
# Output:
#   outputs_stage3/<EPISODE_ID>/<pattern_id>/cutXXXX/pageYYrowZZ/
#     input.png
#     structured.json
#     description.txt
#     meta.json
#     prompt_structured.txt
#     prompt_description.txt
#     raw_structured.txt
#     raw_description.txt

import os
import re
import json
import ast
import glob
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ==================================================
# GLOBAL PARAMS (EDIT HERE)
# ==================================================
APP_ROOT = "/app"
EPISODE_ID = "free/free2"   # ★ stage1 json の episode_id に合わせる（例: "free"）

# Stage1 inputs (FREE dataset)
STAGE1_CUTS_DIR = f"{APP_ROOT}/outputs_free2/free2/cuts"

# Symbol lexicon (id, word[], category, 意味/meaning/...)
SYMBOL_LEXICON_PATH = f"{APP_ROOT}/data/storyboard_symbol_lexicon.json"

# Output root
STAGE3_OUT_ROOT = f"{APP_ROOT}/outputs_stage3/{EPISODE_ID}"
# ==================================================


# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
CFG = {
    # which cuts to process
    "cut_select": "all",   # "all" or "range"
    "cut_start": 1,
    "cut_end": 9999,

    # pattern id (free dataset is picture-only)
    "pattern_id": "free_picture_only",

    # skip if output already exists
    "skip_if_exists": True,

    # image safety
    "max_image_side": 2048,          # resize if max(width,height) > this
    "save_input_image": True,

    # prompts/raw outputs saving
    "save_prompts": True,
    "save_raw_texts": True,

    # model
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,

    # generation
    "gen_struct": {"max_new_tokens": 1400, "do_sample": False, "num_beams": 1},
    "gen_desc":   {"max_new_tokens":  420, "do_sample": False, "num_beams": 1},

    # retries for structured JSON parse
    "max_structured_retries": 2,

    # term injection
    # Inject ONLY terms detected in ordered OCR text (bounded), but meanings MUST be included.
    "max_terms_with_meaning": 50,
    "term_scan_max_chars": 8000,

    # placeholders (remove noise lines)
    "placeholders": ["□", "△"],

    # description style hint (optional)
    "description_style_hint": (
        "Write concise Japanese suitable for a paper appendix. "
        "Prefer objective description of composition, character pose, and action. "
        "If OCR implies progression across rows, keep the temporal order."
    ),

    # output language
    "lang": "ja",
}
# ==================================================


# -------------------------
# Regex / parsing helpers
# -------------------------
CUT_STAGE1_RE = re.compile(r"cut(\d+)\.stage1\.json$")
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

ASCII_TOKEN_RE = re.compile(r"^[A-Za-z0-9\.\-]+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, s: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write((s or "").rstrip() + "\n")


def parse_cut_num_from_stage1_path(path: str) -> Optional[int]:
    m = CUT_STAGE1_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def cut_in_range(cut_num: Optional[int]) -> bool:
    if CFG["cut_select"] == "all":
        return True
    if cut_num is None:
        return False
    return int(CFG["cut_start"]) <= int(cut_num) <= int(CFG["cut_end"])


def resolve_path(p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(APP_ROOT, p)


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_to_max_side(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return im.resize((nw, nh), Image.BICUBIC)


def normalize_placeholders(text: str) -> str:
    """
    Remove placeholder-only lines (□, △) to reduce prompt noise.
    Keep line order otherwise.
    """
    if not isinstance(text, str):
        return ""
    ph = set(CFG.get("placeholders", []))
    lines = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t in ph:
            continue
        lines.append(ln.rstrip())
    return "\n".join(lines).strip()


def get_cell_raw_and_corrected(cell: Dict[str, Any]) -> Tuple[str, str]:
    """
    Stage1 cell has raw_text; Stage2 split cell may have corrected_text.
    For free Stage1-only: corrected == raw.
    """
    if not isinstance(cell, dict):
        return "", ""
    raw = cell.get("raw_text")
    raw_s = raw if isinstance(raw, str) else ""
    corr = cell.get("corrected_text")
    corr_s = corr if isinstance(corr, str) else raw_s
    return normalize_placeholders(raw_s), normalize_placeholders(corr_s)


# -------------------------
# Symbol lexicon: term -> {category, meaning}
# -------------------------
def _meaning_to_string(v: Any) -> str:
    """
    Robust conversion:
    - str -> strip
    - list -> join stringified items
    - dict -> prefer ja/jp/text/value; else json dump
    - other -> str
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for x in v:
            xs = _meaning_to_string(x)
            if xs:
                parts.append(xs)
        return " / ".join(parts).strip()
    if isinstance(v, dict):
        for k in ["ja", "jp", "japanese", "text", "value", "desc", "description", "meaning", "意味"]:
            if k in v:
                s = _meaning_to_string(v.get(k))
                if s:
                    return s
        try:
            return json.dumps(v, ensure_ascii=False).strip()
        except Exception:
            return str(v).strip()
    return str(v).strip()


def _extract_meaning(item: Dict[str, Any]) -> str:
    """
    Try multiple possible keys.
    """
    for k in ["意味", "meaning", "Meaning", "description", "desc", "説明", "備考", "note", "notes"]:
        if k in item:
            s = _meaning_to_string(item.get(k))
            if s:
                return s
    # nested patterns
    if isinstance(item.get("meaning"), dict):
        s = _meaning_to_string(item.get("meaning"))
        if s:
            return s
    return ""


def load_symbol_lexicon_full(path: str) -> Dict[str, Dict[str, str]]:
    """
    JSON entries: {id, word:[...], category, 意味/meaning/...}
    Return term -> {"category": ..., "meaning": ...}

    IMPORTANT: meaning extraction is robust (string/list/dict).
    """
    if not os.path.isfile(path):
        return {}

    data = read_json(path)
    term2: Dict[str, Dict[str, str]] = {}

    if not isinstance(data, list):
        return term2

    for item in data:
        if not isinstance(item, dict):
            continue

        cat = item.get("category", "unknown")
        cat_s = str(cat).strip() if cat is not None else "unknown"

        meaning_s = _extract_meaning(item)

        words = item.get("word", [])
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, list):
            continue

        for w in words:
            if not isinstance(w, str):
                continue
            term = w.strip()
            if not term:
                continue

            if term not in term2:
                term2[term] = {"category": cat_s, "meaning": meaning_s}
            else:
                # prefer non-empty meaning
                if (not term2[term].get("meaning")) and meaning_s:
                    term2[term]["meaning"] = meaning_s
                # prefer non-unknown category
                if (term2[term].get("category") in ["", "unknown"]) and cat_s not in ["", "unknown"]:
                    term2[term]["category"] = cat_s

    return term2


def _ascii_token_boundary_pattern(term: str) -> re.Pattern:
    """
    Match ASCII-like term as a token (avoid matching "S" inside "SE", "F" inside "FIX", etc).
    Boundary definition: not preceded/followed by [A-Za-z0-9]
    """
    esc = re.escape(term)
    return re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])")


def detect_terms_in_text(text: str, term2info: Dict[str, Dict[str, str]], max_k: int) -> List[Dict[str, str]]:
    """
    Detect terms in text.
    - Longest-first
    - ASCII-like terms use boundary match to reduce false positives.
    - Meanings are injected (must be non-empty if available).
    """
    if not text or not term2info:
        return []

    scan = text[: int(CFG["term_scan_max_chars"])]
    terms_sorted = sorted(term2info.keys(), key=lambda x: (-len(x), x))

    out: List[Dict[str, str]] = []
    seen = set()

    for term in terms_sorted:
        if term in seen:
            continue

        hit = False
        if ASCII_TOKEN_RE.match(term):
            # token-boundary match
            if _ascii_token_boundary_pattern(term).search(scan):
                hit = True
        else:
            # simple substring match for JP terms / symbols
            if term in scan:
                hit = True

        if not hit:
            continue

        seen.add(term)
        info = term2info.get(term, {})
        out.append({
            "term": term,
            "category": info.get("category", "unknown"),
            "meaning": info.get("meaning", ""),  # ★ここが空なら lexicon 側が空
        })

        if len(out) >= int(max_k):
            break

    return out


def build_terms_block(terms_with_meaning: List[Dict[str, str]]) -> str:
    if not terms_with_meaning:
        return "(none)"
    lines = []
    for t in terms_with_meaning:
        term = t.get("term", "")
        cat = t.get("category", "unknown")
        meaning = t.get("meaning", "")
        # IMPORTANT: show meaning even if empty (for debugging), but ideally it won't be empty anymore.
        lines.append(f"- {term} ({cat}): {meaning}")
    return "\n".join(lines)


def term_meaning_stats(term2info: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    if not term2info:
        return {"n_terms": 0}
    n = len(term2info)
    n_nonempty = sum(1 for _, v in term2info.items() if (v.get("meaning") or "").strip() != "")
    # some examples of empty meaning terms
    empty_terms = []
    for k, v in term2info.items():
        if (v.get("meaning") or "").strip() == "":
            empty_terms.append(k)
        if len(empty_terms) >= 20:
            break
    return {
        "n_terms": n,
        "n_terms_with_nonempty_meaning": n_nonempty,
        "ratio_nonempty": (n_nonempty / n) if n > 0 else None,
        "example_empty_meaning_terms": empty_terms,
    }


# -------------------------
# Row-order note (EN)
# -------------------------
ROW_ORDER_NOTE_EN = (
    "IMPORTANT ABOUT TEXT ORDER:\n"
    "- The storyboard is organized in ROWS from top to bottom.\n"
    "- The provided text blocks are ORDERED by (page asc, row asc).\n"
    "- Do NOT reorder them. The order is meaningful (temporal / progression / shot sequence).\n"
)


# -------------------------
# Prompts (EN; output JSON / JP description)
# -------------------------
def build_structured_prompt(
    *,
    cut_num: int,
    pattern_id: str,
    unit_obj: Dict[str, Any],
    row_action_block: str,
    row_dialogue_block: str,
    cut_action_block: str,
    cut_dialogue_block: str,
    script_evidence: List[Dict[str, Any]],
    terms_with_meaning: List[Dict[str, str]],
) -> str:
    # script evidence: keep interface; free dataset usually has none
    script_lines = []
    for c in script_evidence:
        script_lines.append(f"[{c.get('chunk_id')}] speaker={c.get('speaker')} text={c.get('text')}")
    script_block = "\n".join(script_lines) if script_lines else "(none)"

    terms_block = build_terms_block(terms_with_meaning)

    prompt = (
        "You analyze an anime storyboard and output STRICT JSON only.\n"
        "You are given an input image and optional text/context.\n\n"
        "CRITICAL PRIORITY:\n"
        "1) The IMAGE is the PRIMARY evidence. You MUST look at the image carefully.\n"
        "2) OCR text and retrieved script are only HINTS.\n"
        "If the text conflicts with the image, prioritize the image and note the conflict.\n\n"
        + ROW_ORDER_NOTE_EN + "\n"
        "Rules:\n"
        "- Output must be STRICT JSON. Use double quotes for all strings.\n"
        "- Do NOT output markdown code fences.\n"
        "- Do NOT invent details that are not supported by the image.\n"
        "- If unknown, use null or empty list.\n"
        "- You MUST include at least 3 short image observations in evidence.image_observations.\n"
        "- Camera movement must be included ONLY if it is visible in the image or explicitly written in OCR.\n"
        "- Use the glossary meanings to interpret production terms (e.g., FIX, SE, PAN, O.L).\n"
        "- Do NOT copy-paste the full glossary meaning into outputs; use it only for understanding.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        "=== TEXT (CURRENT UNIT; order-aware) ===\n"
        "[action_memo_row_raw_and_corrected]\n"
        f"{row_action_block}\n\n"
        "[dialogue_row_raw_and_corrected]\n"
        f"{row_dialogue_block}\n\n"
        "=== TEXT (WHOLE CUT CONTEXT; ORDERED) ===\n"
        "[action_memo_cut_ordered_raw_and_corrected]\n"
        f"{cut_action_block}\n\n"
        "[dialogue_cut_ordered_raw_and_corrected]\n"
        f"{cut_dialogue_block}\n\n"
        "=== SCRIPT (retrieved, hint) ===\n"
        f"{script_block}\n\n"
        "=== TERMS (meaning attached, hint) ===\n"
        f"{terms_block}\n\n"
        "=== OUTPUT JSON SCHEMA ===\n"
        "{\n"
        '  "cut": <int>,\n'
        '  "pattern": "<pattern_id>",\n'
        '  "unit": { ... },\n'
        '  "scene": {\n'
        '    "setting": <string|null>,\n'
        '    "characters": <array of strings>,\n'
        '    "actions": <array of strings>,\n'
        '    "camera": {"movement": <array of strings>, "framing": <string|null>, "focus": <string|null>},\n'
        '    "sound": <array of {"type": "SE|BGM|GAYA|other", "text": string}>,\n'
        '    "dialogue": <array of {"speaker": string|null, "text": string}>\n'
        "  },\n"
        '  "consistency": {"script_match": "match|partial|unknown|mismatch", "notes": <array of strings>},\n'
        '  "evidence": {\n'
        '    "image_observations": <array of strings>,\n'
        '    "used_script_chunk_ids": <array of strings>,\n'
        '    "used_terms_with_meaning": <array of {"term": string, "meaning": string}>,\n'
        '    "inputs": {\n'
        '       "ocr_action_memo_row": string,\n'
        '       "ocr_dialogue_row": string,\n'
        '       "ocr_action_memo_cut": string,\n'
        '       "ocr_dialogue_cut": string,\n'
        '       "script_snippets": <array of strings>\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        f"Set cut={cut_num} and pattern='{pattern_id}'.\n"
        "Return JSON only.\n"
    )
    return prompt


def build_description_prompt(
    *,
    cut_num: int,
    pattern_id: str,
    unit_obj: Dict[str, Any],
    row_action_block: str,
    row_dialogue_block: str,
    description_style_hint: str,
    structured_json: Dict[str, Any],
    terms_with_meaning: List[Dict[str, str]],
) -> str:
    terms_block = build_terms_block(terms_with_meaning)

    prompt = (
        "You write a short Japanese scene description for an anime storyboard.\n"
        "You are given an input image, ordered text blocks, and a structured JSON analysis.\n\n"
        "CRITICAL PRIORITY:\n"
        "1) The IMAGE is the PRIMARY evidence. You MUST look at the image carefully.\n"
        "2) The structured JSON is only a checklist to avoid missing points.\n"
        "If the JSON conflicts with the image, prioritize the image and ignore the conflicting JSON parts.\n\n"
        + ROW_ORDER_NOTE_EN + "\n"
        "Rules:\n"
        "- Do NOT invent details.\n"
        "- Output MUST be 2-4 sentences in Japanese.\n"
        "- Include at least TWO concrete visual facts from the image.\n"
        "- Mention camera movement only if it is visible in the image or explicitly written in OCR.\n"
        "- Dialogue should be included only if supported by OCR.\n"
        "- Keep it consistent with the row order if the text suggests progression.\n"
        "- Use the glossary meanings only to interpret production terms; do not copy them verbatim.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        "=== TEXT (CURRENT UNIT; order-aware) ===\n"
        "[action_memo_row_raw_and_corrected]\n"
        f"{row_action_block}\n\n"
        "[dialogue_row_raw_and_corrected]\n"
        f"{row_dialogue_block}\n\n"
        "=== TERMS (meaning attached, hint) ===\n"
        f"{terms_block}\n\n"
        f"=== DESCRIPTION STYLE HINT ===\n{description_style_hint}\n\n"
        f"=== STRUCTURED JSON (checklist) ===\n{json.dumps(structured_json, ensure_ascii=False)}\n\n"
        "=== OUTPUT ===\nReturn ONLY the Japanese description text.\n"
    )
    return prompt


# -------------------------
# Model helpers
# -------------------------
def load_model():
    kwargs = {"device_map": CFG["device_map"], "torch_dtype": CFG["torch_dtype"]}
    if CFG["flash_attn2"]:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(CFG["model_id"], **kwargs)
    processor = AutoProcessor.from_pretrained(CFG["model_id"])
    model.eval()

    tok = processor.tokenizer
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return model, processor


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return x
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 3 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def _move_and_fix_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    if "input_ids" in out and isinstance(out["input_ids"], torch.Tensor):
        out["input_ids"] = _ensure_2d(out["input_ids"])
    if "attention_mask" in out and isinstance(out["attention_mask"], torch.Tensor):
        out["attention_mask"] = _ensure_2d(out["attention_mask"])
    return out


@torch.no_grad()
def vl_generate_text(model, processor, image: Image.Image, prompt: str, gen_cfg: Dict[str, Any]) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = _move_and_fix_batch(inputs, model.device)

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    g = dict(gen_cfg)
    g["pad_token_id"] = pad_id
    if eos_id is not None:
        g["eos_token_id"] = eos_id

    out_ids = model.generate(**inputs, **g)
    trimmed = out_ids[0][len(inputs["input_ids"][0]):]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


# -------------------------
# Robust JSON parsing
# -------------------------
def _strip_code_fences(text: str) -> str:
    m = CODE_FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def _extract_brace_block(text: str) -> str:
    s = (text or "")
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j+1]
    return s.strip()


def _remove_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def parse_json_relaxed(text: str) -> Dict[str, Any]:
    """
    strict JSON -> repair -> ast.literal_eval fallback
    """
    t = _strip_code_fences(text)
    t = _extract_brace_block(t)
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    t2 = _remove_trailing_commas(t)

    try:
        return json.loads(t2)
    except Exception:
        pass

    t3 = t2
    t3 = re.sub(r"\bnull\b", "None", t3, flags=re.IGNORECASE)
    t3 = re.sub(r"\btrue\b", "True", t3, flags=re.IGNORECASE)
    t3 = re.sub(r"\bfalse\b", "False", t3, flags=re.IGNORECASE)

    obj = ast.literal_eval(t3)
    if isinstance(obj, dict):
        return obj
    raise json.JSONDecodeError("Failed to parse JSON (relaxed)", text, 0)


def normalize_structured_schema(obj: Dict[str, Any], cut_num: int, pattern_id: str, unit_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    out = dict(obj)

    out["cut"] = int(out.get("cut", cut_num) or cut_num)
    out["pattern"] = str(out.get("pattern", pattern_id) or pattern_id)
    out["unit"] = out.get("unit", unit_obj)
    if not isinstance(out["unit"], dict):
        out["unit"] = unit_obj

    scene = out.get("scene")
    if not isinstance(scene, dict):
        scene = {}
    scene.setdefault("setting", None)
    scene.setdefault("characters", [])
    scene.setdefault("actions", [])
    cam = scene.get("camera")
    if not isinstance(cam, dict):
        cam = {}
    cam.setdefault("movement", [])
    cam.setdefault("framing", None)
    cam.setdefault("focus", None)
    scene["camera"] = cam
    scene.setdefault("sound", [])
    scene.setdefault("dialogue", [])
    out["scene"] = scene

    cons = out.get("consistency")
    if not isinstance(cons, dict):
        cons = {}
    cons.setdefault("script_match", "unknown")
    cons.setdefault("notes", [])
    out["consistency"] = cons

    ev = out.get("evidence")
    if not isinstance(ev, dict):
        ev = {}
    ev.setdefault("image_observations", [])
    ev.setdefault("used_script_chunk_ids", [])
    ev.setdefault("used_terms_with_meaning", [])
    inputs = ev.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    inputs.setdefault("ocr_action_memo_row", "")
    inputs.setdefault("ocr_dialogue_row", "")
    inputs.setdefault("ocr_action_memo_cut", "")
    inputs.setdefault("ocr_dialogue_cut", "")
    inputs.setdefault("script_snippets", [])
    ev["inputs"] = inputs
    out["evidence"] = ev

    return out


# -------------------------
# Text block builders (raw+corrected; ordered)
# -------------------------
def format_raw_corrected_block(raw: str, corr: str) -> str:
    raw = raw or ""
    corr = corr or ""
    if not raw and not corr:
        return "(empty)"
    return f"RAW:\n{raw if raw else '(empty)'}\n\nCORRECTED:\n{corr if corr else '(empty)'}"


def build_cut_ordered_block(rows_sorted: List[Dict[str, Any]], col_name: str) -> str:
    parts = []
    for r in rows_sorted:
        page = r.get("page")
        row = r.get("row")
        cols = r.get("cols", {}) or {}
        cell = cols.get(col_name, {}) or {}
        raw, corr = get_cell_raw_and_corrected(cell)
        if not raw and not corr:
            continue
        parts.append(f"[page={int(page):02d} row={int(row):02d}]\n{format_raw_corrected_block(raw, corr)}")
    return "\n\n".join(parts).strip() if parts else "(none)"


def build_row_block(row_obj: Dict[str, Any], col_name: str) -> str:
    cols = row_obj.get("cols", {}) or {}
    cell = cols.get(col_name, {}) or {}
    raw, corr = get_cell_raw_and_corrected(cell)
    return format_raw_corrected_block(raw, corr)


def collect_text_for_term_detection(row_action_block: str, row_dialogue_block: str, cut_action_block: str, cut_dialogue_block: str) -> str:
    return "\n".join([row_action_block, row_dialogue_block, cut_action_block, cut_dialogue_block]).strip()


# -------------------------
# Save bundle
# -------------------------
def save_bundle(
    out_dir: str,
    *,
    image: Optional[Image.Image],
    structured: Dict[str, Any],
    description: str,
    meta: Dict[str, Any],
    prompt_struct: str,
    prompt_desc: str,
    raw_struct: str,
    raw_desc: str,
):
    ensure_dir(out_dir)

    if CFG.get("save_input_image", True) and image is not None:
        image.save(os.path.join(out_dir, "input.png"))

    write_json(os.path.join(out_dir, "structured.json"), structured)
    write_text(os.path.join(out_dir, "description.txt"), description)
    write_json(os.path.join(out_dir, "meta.json"), meta)

    if CFG.get("save_prompts", True):
        write_text(os.path.join(out_dir, "prompt_structured.txt"), prompt_struct)
        write_text(os.path.join(out_dir, "prompt_description.txt"), prompt_desc)

    if CFG.get("save_raw_texts", True):
        write_text(os.path.join(out_dir, "raw_structured.txt"), raw_struct)
        write_text(os.path.join(out_dir, "raw_description.txt"), raw_desc)


# -------------------------
# Main
# -------------------------
def main():
    ensure_dir(STAGE3_OUT_ROOT)

    # Load term meanings (must be included)
    term2info = load_symbol_lexicon_full(SYMBOL_LEXICON_PATH)
    stats = term_meaning_stats(term2info)
    print("[INFO] symbol lexicon stats:", stats)
    if stats.get("n_terms", 0) > 0 and (stats.get("ratio_nonempty") is not None) and stats["ratio_nonempty"] < 0.5:
        print("[WARN] Many terms have empty meanings. Check lexicon keys (意味/meaning/etc). Examples:", stats.get("example_empty_meaning_terms"))

    # Load model once
    model, processor = load_model()

    # List stage1 cut files
    cut_files = sorted(glob.glob(os.path.join(STAGE1_CUTS_DIR, "cut*.stage1.json")))
    if not cut_files:
        raise FileNotFoundError(f"No stage1 cut files found: {STAGE1_CUTS_DIR}")

    pattern_id = str(CFG["pattern_id"])

    for s1_path in tqdm(cut_files, desc="Stage3 FREE: Cuts"):
        cut_num = parse_cut_num_from_stage1_path(s1_path)
        if not cut_in_range(cut_num):
            continue

        stage1 = read_json(s1_path)
        cut_num_obj = stage1.get("cut")
        if cut_num_obj is not None:
            try:
                cut_num = int(cut_num_obj)
            except Exception:
                pass
        if cut_num is None:
            cut_num = -1

        rows = stage1.get("rows", []) or []
        if not rows:
            continue

        # Ensure order by (page asc, row asc) (stable)
        rows_sorted = sorted(
            list(enumerate(rows)),
            key=lambda x: (
                int((x[1] or {}).get("page", 10**9)),
                int((x[1] or {}).get("row", 10**9)),
                x[0],
            )
        )
        rows_sorted = [r for _, r in rows_sorted]

        # Build cut-level ordered text blocks
        cut_action_block = build_cut_ordered_block(rows_sorted, "action_memo")
        cut_dialogue_block = build_cut_ordered_block(rows_sorted, "dialogue")

        cut_dirname = f"cut{int(cut_num):04d}" if cut_num >= 0 else "cut_unknown"

        for row_obj in tqdm(rows_sorted, desc=f"Cut {cut_num}: Rows", leave=False):
            try:
                page = int(row_obj.get("page", -1))
                row = int(row_obj.get("row", -1))
            except Exception:
                continue
            if page < 0 or row < 0:
                continue

            unit_obj = {"page": page, "row": row}
            unit_folder = f"page{page:02d}row{row:02d}"

            # picture image path (IMPORTANT: each row has its own picture)
            pic_cell = (row_obj.get("cols", {}) or {}).get("picture", {}) or {}
            pic_rel = pic_cell.get("image")
            if not isinstance(pic_rel, str) or not pic_rel.strip():
                continue
            pic_path = resolve_path(pic_rel.strip())
            if not os.path.isfile(pic_path):
                continue

            out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname, unit_folder)
            if CFG.get("skip_if_exists", True) and os.path.isfile(os.path.join(out_dir, "description.txt")):
                continue

            # Load image (picture only)
            img = open_rgb(pic_path)
            img = resize_to_max_side(img, int(CFG["max_image_side"]))

            # Build row blocks (raw+corrected; free stage1 -> corrected==raw)
            row_action_block = build_row_block(row_obj, "action_memo")
            row_dialogue_block = build_row_block(row_obj, "dialogue")

            # Detect terms in text blocks, inject meanings
            text_for_terms = collect_text_for_term_detection(
                row_action_block=row_action_block,
                row_dialogue_block=row_dialogue_block,
                cut_action_block=cut_action_block,
                cut_dialogue_block=cut_dialogue_block,
            )
            terms_with_meaning = detect_terms_in_text(
                text=text_for_terms,
                term2info=term2info,
                max_k=int(CFG["max_terms_with_meaning"]),
            )

            # Free dataset: no script retrieval by default
            script_evidence: List[Dict[str, Any]] = []

            # -------- structured --------
            prompt_struct = build_structured_prompt(
                cut_num=cut_num,
                pattern_id=pattern_id,
                unit_obj=unit_obj,
                row_action_block=row_action_block,
                row_dialogue_block=row_dialogue_block,
                cut_action_block=cut_action_block,
                cut_dialogue_block=cut_dialogue_block,
                script_evidence=script_evidence,
                terms_with_meaning=terms_with_meaning,
            )

            raw_struct = ""
            structured: Optional[Dict[str, Any]] = None
            last_err = None

            for _ in range(int(CFG["max_structured_retries"]) + 1):
                raw_struct = vl_generate_text(model, processor, img, prompt_struct, CFG["gen_struct"])
                try:
                    parsed = parse_json_relaxed(raw_struct)
                    structured = normalize_structured_schema(parsed, cut_num, pattern_id, unit_obj)
                    break
                except Exception as e:
                    last_err = str(e)
                    prompt_struct = prompt_struct + "\nREMINDER: Output STRICT JSON ONLY. No extra text.\n"

            if structured is None:
                structured = normalize_structured_schema({}, cut_num, pattern_id, unit_obj)
                structured.setdefault("consistency", {})
                structured["consistency"].setdefault("notes", [])
                structured["consistency"]["notes"].append(f"structured_parse_failed: {last_err}")

            # -------- description --------
            prompt_desc = build_description_prompt(
                cut_num=cut_num,
                pattern_id=pattern_id,
                unit_obj=unit_obj,
                row_action_block=row_action_block,
                row_dialogue_block=row_dialogue_block,
                description_style_hint=str(CFG.get("description_style_hint", "")),
                structured_json=structured,
                terms_with_meaning=terms_with_meaning,
            )
            raw_desc = vl_generate_text(model, processor, img, prompt_desc, CFG["gen_desc"])
            description = raw_desc.strip()

            meta = {
                "created_at": now_iso(),
                "episode_id": EPISODE_ID,
                "cut": cut_num,
                "pattern": pattern_id,
                "unit": unit_obj,
                "stage1_path": s1_path,
                "picture_path": pic_path,
                "model_id": CFG["model_id"],
                "gen_struct": CFG["gen_struct"],
                "gen_desc": CFG["gen_desc"],
                "note": "FREE dataset: multiple rows per cut; picture-only image; ordered row/cut text blocks; term meanings injected.",
                "n_terms_injected": len(terms_with_meaning),
                "terms_injected": [t.get("term") for t in terms_with_meaning],
            }

            save_bundle(
                out_dir,
                image=img,
                structured=structured,
                description=description,
                meta=meta,
                prompt_struct=prompt_struct,
                prompt_desc=prompt_desc,
                raw_struct=raw_struct,
                raw_desc=raw_desc,
            )


if __name__ == "__main__":
    main()

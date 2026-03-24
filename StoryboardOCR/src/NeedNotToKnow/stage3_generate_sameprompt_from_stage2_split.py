# stage3_generate_sameprompt_from_stage2_split.py
#
# Stage3:
# - Input image varies by pattern:
#     p1_row_picture : picture crop only
#     p2_row_concat  : row concat image (cut|picture|action_memo|dialogue|time)
#     p3_cut_concat  : cut concat image (stack row concat images)
#
# - IMPORTANT: Prompt template is the SAME for all patterns.
#   (pattern_id is recorded in meta and also asked in JSON, but instruction text is not pattern-specific)
#
# Stage2 input (NEW):
#   outputs_stage2_clm/episode01/cuts_split/cutXXXX.stage2.split.(nojp|jp).json
#
# We provide text evidence as:
#   - CURRENT ROW raw/corrected (primary for row patterns)
#   - WHOLE CUT ordered raw/corrected (context, and order is emphasized)
#
# Outputs:
#   outputs_stage3/<episode01>/<pattern>/cutXXXX/.../
#     input.png
#     structured.json
#     description.txt
#     meta.json
#     raw_structured.txt
#     raw_description.txt
#     prompt_structured.txt
#     prompt_description.txt

import os
import re
import json
import ast
from tqdm import tqdm
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ==================================================
# GLOBAL PARAMS (EDIT HERE)
# ==================================================
APP_ROOT = "/app"
EPISODE_ID = "episode01"

STAGE1_CUTS_DIR = f"{APP_ROOT}/outputs/{EPISODE_ID}/cuts"

# NEW: stage2 split dir
STAGE2_SPLIT_DIR = f"{APP_ROOT}/outputs_stage2_clm/{EPISODE_ID}/cuts_split"

SYMBOL_LEXICON_PATH = f"{APP_ROOT}/data/storyboard_symbol_lexicon.json"
STAGE3_OUT_ROOT = f"{APP_ROOT}/outputs_stage3/{EPISODE_ID}"
# ==================================================


# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
CFG = {
    # which cuts to process
    "cut_select": "all",   # "all" or "range"
    "cut_start": 1,
    "cut_end": 10,

    # patterns to run
    "patterns": ["p1_row_picture", "p2_row_concat", "p3_cut_concat"],

    # Stage2 variant to use ("nojp" or "jp")
    "stage2_variant": "nojp",

    # skip if output already exists
    "skip_if_exists": True,

    # concat layout
    "row_concat_order": ["cut", "picture", "action_memo", "dialogue", "time"],  # left->right
    "row_concat_pad": 10,
    "row_concat_bg": (255, 255, 255),
    "row_placeholder_size": (220, 220),

    "cut_concat_pad": 10,
    "cut_concat_bg": (255, 255, 255),

    # image safety (important for p3)
    "max_image_side": 2048,   # resize if max(width,height) > this
    "save_input_image": True,

    # model
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,

    # generation
    "gen_struct": {"max_new_tokens": 1400, "do_sample": False, "num_beams": 1},
    "gen_desc":   {"max_new_tokens":  420, "do_sample": False, "num_beams": 1},

    # RAG injection sizes (from Stage2 rag block)
    "max_script_evidence": 3,
    "max_terms_with_meaning": 25,

    # output saving
    "save_prompts": True,
    "save_raw_texts": True,

    # retries for structured JSON parse
    "max_structured_retries": 2,

    # output language
    "lang": "ja",

    # Text length safety (optional; avoids extremely long prompts)
    "max_cut_text_chars": 4000,
}
# ==================================================


# -------------------------
# File helpers
# -------------------------
S1_FILE_RE = re.compile(r"cut(\d+)\.stage1\.json$")
S2_SPLIT_RE = re.compile(r"cut(\d+)\.stage2\.split\.(nojp|jp)\.json$")
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

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

def parse_cut_num_stage1(path: str) -> Optional[int]:
    m = S1_FILE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def parse_cut_num_stage2_split(path: str) -> Optional[int]:
    m = S2_SPLIT_RE.search(os.path.basename(path))
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

def hconcat(images: List[Image.Image], pad: int, bg=(255,255,255)) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    H = max(heights) if heights else 1
    W = sum(widths) + pad * (len(images) - 1) if images else 1
    canvas = Image.new("RGB", (W, H), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + pad
    return canvas

def vconcat(images: List[Image.Image], pad: int, bg=(255,255,255)) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    W = max(widths) if widths else 1
    H = sum(heights) + pad * (len(images) - 1) if images else 1
    canvas = Image.new("RGB", (W, H), bg)
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + pad
    return canvas


# -------------------------
# Stage2 split mapping
# -------------------------
def build_stage2_split_map(stage2_split_dir: str, variant: str) -> Dict[int, str]:
    """
    Map: cut_num -> stage2 split path (for selected variant)
    If duplicates exist, pick newest by mtime.
    """
    mp: Dict[int, str] = {}
    mt: Dict[int, float] = {}
    if not os.path.isdir(stage2_split_dir):
        return mp

    for fn in os.listdir(stage2_split_dir):
        m = S2_SPLIT_RE.match(fn)
        if not m:
            continue
        cut_i = int(m.group(1))
        v = m.group(2)
        if v != variant:
            continue
        p = os.path.join(stage2_split_dir, fn)
        t = os.path.getmtime(p)
        if (cut_i not in mp) or (t > mt.get(cut_i, -1.0)):
            mp[cut_i] = p
            mt[cut_i] = t
    return mp


def build_stage1_map(stage1_dir: str) -> Dict[int, str]:
    mp: Dict[int, str] = {}
    mt: Dict[int, float] = {}
    if not os.path.isdir(stage1_dir):
        return mp
    for fn in os.listdir(stage1_dir):
        m = S1_FILE_RE.match(fn)
        if not m:
            continue
        cut_i = int(m.group(1))
        p = os.path.join(stage1_dir, fn)
        t = os.path.getmtime(p)
        if (cut_i not in mp) or (t > mt.get(cut_i, -1.0)):
            mp[cut_i] = p
            mt[cut_i] = t
    return mp


# -------------------------
# Lexicon helpers
# -------------------------
def load_symbol_lexicon(path: str) -> Dict[str, str]:
    """
    Build map: term -> meaning
    JSON entries: {id, word:[...], category, 意味}
    """
    if not os.path.isfile(path):
        return {}
    data = read_json(path)
    term2meaning: Dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            meaning = item.get("意味", "")
            words = item.get("word", [])
            if not isinstance(words, list):
                continue
            for w in words:
                if isinstance(w, str) and w.strip():
                    term2meaning[w.strip()] = meaning if isinstance(meaning, str) else ""
    return term2meaning


# -------------------------
# Stage2 split: row/cut ordered texts + RAG extraction
# -------------------------
def _sorted_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r):
        try:
            return (int(r.get("page", 10**9)), int(r.get("row", 10**9)))
        except Exception:
            return (10**9, 10**9)
    return sorted(rows or [], key=key)

def _get_cell_text(stage2_row: Dict[str, Any], col: str, key: str) -> str:
    cols = stage2_row.get("cols", {}) or {}
    cell = cols.get(col, {}) or {}
    val = cell.get(key, "")
    return str(val) if isinstance(val, str) else ""

def make_ordered_cut_text_block(
    stage2_split: Dict[str, Any],
    *,
    col: str,
    key: str,  # "raw_text" or "corrected_text"
    max_chars: int,
) -> str:
    """
    Build ordered text for the whole cut with explicit row order.
    Example:
      [page=2 row=1]
      ...

      [page=2 row=2]
      ...
    """
    rows = _sorted_rows(stage2_split.get("rows", []) or [])
    parts: List[str] = []
    for r in rows:
        try:
            page = int(r.get("page"))
            row = int(r.get("row"))
        except Exception:
            continue
        t = _get_cell_text(r, col, key)
        if not t.strip():
            continue
        parts.append(f"[page={page} row={row}]\n{t.strip()}")
    out = "\n\n".join(parts).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "\n...(truncated)..."
    return out if out else "(none)"

def find_stage2_row(stage2_split: Dict[str, Any], page: int, row: int) -> Optional[Dict[str, Any]]:
    for r in stage2_split.get("rows", []) or []:
        try:
            if int(r.get("page")) == int(page) and int(r.get("row")) == int(row):
                return r
        except Exception:
            continue
    return None

def get_stage2_rag_blocks(
    stage2_split: Dict[str, Any],
    term2meaning: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Stage2 split rag format (v2):
      rag.script_candidates: [{chunk_id, speaker, text, score, ...}, ...]
      rag.terms_with_meaning: [{term, category, meaning}, ...]
    """
    rag = stage2_split.get("rag", {}) or {}

    # script candidates
    sc = rag.get("script_candidates") or rag.get("script_evidence") or []
    if not isinstance(sc, list):
        sc = []
    script_evidence = sc[: int(CFG["max_script_evidence"])]

    # terms with meaning
    twm = rag.get("terms_with_meaning") or []
    terms_with_meaning: List[Dict[str, str]] = []
    if isinstance(twm, list):
        for x in twm:
            if not isinstance(x, dict):
                continue
            term = str(x.get("term", "") or "").strip()
            if not term:
                continue
            meaning = x.get("meaning")
            if not isinstance(meaning, str) or not meaning.strip():
                meaning = term2meaning.get(term, "")
            terms_with_meaning.append({"term": term, "meaning": meaning or ""})
    terms_with_meaning = terms_with_meaning[: int(CFG["max_terms_with_meaning"])]

    return script_evidence, terms_with_meaning


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
# Prompts (SAME template for all patterns)
# -------------------------
ROW_ORDER_NOTE = (
    "IMPORTANT ABOUT TEXT ORDER:\n"
    "- The storyboard is organized in ROWS from top to bottom.\n"
    "- The provided text blocks are ORDERED by (page asc, row asc).\n"
    "- Do NOT reorder them. The order is meaningful (temporal / progression / shot sequence).\n"
)

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
    script_lines = []
    for c in script_evidence:
        script_lines.append(f"[{c.get('chunk_id')}] speaker={c.get('speaker')} text={c.get('text')}")
    script_block = "\n".join(script_lines) if script_lines else "(none)"

    terms_block = "\n".join([f"- {t['term']}: {t['meaning']}" for t in terms_with_meaning]) if terms_with_meaning else "(none)"

    prompt = (
        "You analyze an anime storyboard and output STRICT JSON only.\n"
        "You are given an input image and optional text/context.\n\n"
        "CRITICAL PRIORITY:\n"
        "1) The IMAGE is the PRIMARY evidence. You MUST look at the image carefully.\n"
        "2) OCR text and retrieved script are only HINTS.\n"
        "If the text conflicts with the image, prioritize the image and note the conflict.\n\n"
        + ROW_ORDER_NOTE + "\n"
        "Rules:\n"
        "- Output must be STRICT JSON. Use double quotes for all strings.\n"
        "- Do NOT output markdown code fences.\n"
        "- Do NOT invent details that are not supported by the image.\n"
        "- If unknown, use null or empty list.\n"
        "- You MUST include at least 3 short image observations in evidence.image_observations.\n"
        "- Camera movement must be included ONLY if it is visible in the image or explicitly written in OCR.\n\n"
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
) -> str:
    prompt = (
        "You write a short Japanese scene description for an anime storyboard.\n"
        "You are given an input image, ordered text blocks, and a structured JSON analysis.\n\n"
        "CRITICAL PRIORITY:\n"
        "1) The IMAGE is the PRIMARY evidence. You MUST look at the image carefully.\n"
        "2) The structured JSON is only a checklist to avoid missing points.\n"
        "If the JSON conflicts with the image, prioritize the image and ignore the conflicting JSON parts.\n\n"
        + ROW_ORDER_NOTE + "\n"
        "Rules:\n"
        "- Do NOT invent details.\n"
        "- 2-4 sentences in Japanese.\n"
        "- Include at least one concrete visual fact from the image.\n"
        "- Mention camera movement only if it is visible in the image or explicitly written in OCR.\n"
        "- Dialogue should be included only if supported by OCR.\n"
        "- Keep it consistent with the row order if the text suggests progression.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        "=== TEXT (CURRENT UNIT; order-aware) ===\n"
        "[action_memo_row_raw_and_corrected]\n"
        f"{row_action_block}\n\n"
        "[dialogue_row_raw_and_corrected]\n"
        f"{row_dialogue_block}\n\n"
        f"=== DESCRIPTION STYLE HINT ===\n{description_style_hint}\n\n"
        f"=== STRUCTURED JSON (checklist) ===\n{json.dumps(structured_json, ensure_ascii=False)}\n\n"
        "=== OUTPUT ===\nReturn ONLY the description text.\n"
    )
    return prompt


# -------------------------
# Images for patterns
# -------------------------
def build_row_images_from_stage1_row(row_obj: Dict[str, Any]) -> Dict[str, str]:
    cols = row_obj.get("cols", {}) or {}
    img_paths = {}
    for k in CFG["row_concat_order"]:
        cell = cols.get(k, {}) or {}
        p = cell.get("image")
        img_paths[k] = resolve_path(p) if isinstance(p, str) and p else ""
    return img_paths

def make_pattern1_image(stage1_row: Dict[str, Any]) -> Optional[Image.Image]:
    cols = stage1_row.get("cols", {}) or {}
    pic = cols.get("picture", {}) or {}
    p = pic.get("image")
    if not isinstance(p, str) or not p:
        return None
    p = resolve_path(p)
    if not os.path.isfile(p):
        return None
    im = open_rgb(p)
    return resize_to_max_side(im, int(CFG["max_image_side"]))

def make_pattern2_image(stage1_row: Dict[str, Any]) -> Optional[Image.Image]:
    paths = build_row_images_from_stage1_row(stage1_row)
    ims = []
    for k in CFG["row_concat_order"]:
        p = paths.get(k, "")
        if p and os.path.isfile(p):
            im = open_rgb(p)
        else:
            im = Image.new("RGB", tuple(CFG["row_placeholder_size"]), CFG["row_concat_bg"])
        ims.append(im)
    out = hconcat(ims, int(CFG["row_concat_pad"]), CFG["row_concat_bg"])
    return resize_to_max_side(out, int(CFG["max_image_side"]))

def make_pattern3_image(stage1_cut: Dict[str, Any]) -> Optional[Image.Image]:
    row_imgs = []
    for row_obj in (stage1_cut.get("rows", []) or []):
        im = make_pattern2_image(row_obj)
        if im is not None:
            row_imgs.append(im)
    if not row_imgs:
        return None
    out = vconcat(row_imgs, int(CFG["cut_concat_pad"]), CFG["cut_concat_bg"])
    return resize_to_max_side(out, int(CFG["max_image_side"]))


# -------------------------
# Text blocks for prompts (row-unit + cut-context)
# -------------------------
def build_row_text_block(stage2_row: Optional[Dict[str, Any]], col: str) -> str:
    if stage2_row is None:
        return "(none)"
    raw = _get_cell_text(stage2_row, col, "raw_text").strip()
    cor = _get_cell_text(stage2_row, col, "corrected_text").strip()
    if not raw and not cor:
        return "(none)"
    return (
        "(RAW OCR)\n"
        f"{raw if raw else '(none)'}\n\n"
        "(CORRECTED)\n"
        f"{cor if cor else '(none)'}"
    )

def build_cut_text_block(stage2_split: Dict[str, Any], col: str) -> str:
    raw_block = make_ordered_cut_text_block(
        stage2_split, col=col, key="raw_text",
        max_chars=int(CFG["max_cut_text_chars"])
    )
    cor_block = make_ordered_cut_text_block(
        stage2_split, col=col, key="corrected_text",
        max_chars=int(CFG["max_cut_text_chars"])
    )
    return (
        "(ORDERED RAW OCR)\n"
        f"{raw_block}\n\n"
        "(ORDERED CORRECTED)\n"
        f"{cor_block}"
    )


# -------------------------
# Save bundle
# -------------------------
def save_bundle(
    base_dir: str,
    *,
    image: Optional[Image.Image],
    structured: Dict[str, Any],
    description: str,
    meta: Dict[str, Any],
    raw_structured_text: str,
    raw_description_text: str,
    prompt_struct: str,
    prompt_desc: str,
):
    ensure_dir(base_dir)

    if CFG["save_input_image"] and image is not None:
        image.save(os.path.join(base_dir, "input.png"))

    write_json(os.path.join(base_dir, "structured.json"), structured)
    write_text(os.path.join(base_dir, "description.txt"), description)
    write_json(os.path.join(base_dir, "meta.json"), meta)

    if CFG.get("save_raw_texts", True):
        write_text(os.path.join(base_dir, "raw_structured.txt"), raw_structured_text)
        write_text(os.path.join(base_dir, "raw_description.txt"), raw_description_text)

    if CFG.get("save_prompts", True):
        write_text(os.path.join(base_dir, "prompt_structured.txt"), prompt_struct)
        write_text(os.path.join(base_dir, "prompt_description.txt"), prompt_desc)


# -------------------------
# Main
# -------------------------
def main():
    ensure_dir(STAGE3_OUT_ROOT)

    term2meaning = load_symbol_lexicon(SYMBOL_LEXICON_PATH)
    model, processor = load_model()

    stage2_variant = str(CFG["stage2_variant"])
    if stage2_variant not in ("nojp", "jp"):
        raise ValueError("CFG['stage2_variant'] must be 'nojp' or 'jp'")

    s2_map = build_stage2_split_map(STAGE2_SPLIT_DIR, variant=stage2_variant)
    s1_map = build_stage1_map(STAGE1_CUTS_DIR)

    cut_nums = sorted([c for c in s2_map.keys() if cut_in_range(c)])
    if not cut_nums:
        raise FileNotFoundError(f"No stage2 split files found for variant={stage2_variant} in: {STAGE2_SPLIT_DIR}")

    for cut_num in tqdm(cut_nums, desc="Stage3: Cuts"):
        s2_path = s2_map.get(cut_num)
        s1_path = s1_map.get(cut_num)

        if not s2_path or not os.path.isfile(s2_path):
            continue
        if not s1_path or not os.path.isfile(s1_path):
            continue

        stage2_split = read_json(s2_path)
        stage1_cut = read_json(s1_path)

        # RAG blocks (hint)
        script_evidence, terms_with_meaning = get_stage2_rag_blocks(stage2_split, term2meaning)

        # Cut context (ORDERED raw + corrected)
        cut_action_block = build_cut_text_block(stage2_split, "action_memo")
        cut_dialogue_block = build_cut_text_block(stage2_split, "dialogue")

        cut_dirname = f"cut{int(cut_num):04d}"

        # build index for stage2 rows
        stage2_rows = stage2_split.get("rows", []) or []
        stage2_rows_sorted = _sorted_rows(stage2_rows)

        # Stage1 rows for images (also ordered)
        stage1_rows = stage1_cut.get("rows", []) or []
        stage1_rows_sorted = _sorted_rows(stage1_rows)

        # Map (page,row)->stage1_row for image building
        s1_row_map: Dict[Tuple[int,int], Dict[str,Any]] = {}
        for r in stage1_rows_sorted:
            try:
                s1_row_map[(int(r.get("page")), int(r.get("row")))] = r
            except Exception:
                continue

        # -------------------------
        # Pattern 1 / 2: row-based
        # -------------------------
        if ("p1_row_picture" in CFG["patterns"]) or ("p2_row_concat" in CFG["patterns"]):
            for s2_row in tqdm(stage2_rows_sorted, desc=f"Cut {cut_num}: Rows", leave=False):
                try:
                    page = int(s2_row.get("page", -1))
                    row = int(s2_row.get("row", -1))
                except Exception:
                    continue
                if page < 0 or row < 0:
                    continue

                stage1_row = s1_row_map.get((page, row))
                if stage1_row is None:
                    continue

                unit_folder = f"page{page:02d}row{row:02d}"
                unit_obj = {"page": page, "row": row}

                # CURRENT ROW raw+corrected blocks (primary for row patterns)
                row_action_block = build_row_text_block(s2_row, "action_memo")
                row_dialogue_block = build_row_text_block(s2_row, "dialogue")

                # style hint (kept same across patterns; just a small hint)
                style_hint = "2〜4文で、絵（画像）を根拠に具体的に。テキストは補助。"

                # Pattern1: picture image only
                if "p1_row_picture" in CFG["patterns"]:
                    pattern_id = "p1_row_picture"
                    out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname, unit_folder)
                    if (not CFG["skip_if_exists"]) or (not os.path.isfile(os.path.join(out_dir, "description.txt"))):
                        img = make_pattern1_image(stage1_row)
                        if img is not None:
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

                            raw_structured = ""
                            structured = None
                            last_err = None
                            for _ in range(int(CFG["max_structured_retries"]) + 1):
                                raw_structured = vl_generate_text(model, processor, img, prompt_struct, CFG["gen_struct"])
                                try:
                                    parsed = parse_json_relaxed(raw_structured)
                                    structured = normalize_structured_schema(parsed, cut_num, pattern_id, unit_obj)
                                    break
                                except Exception as e:
                                    last_err = str(e)
                                    prompt_struct = prompt_struct + "\nREMINDER: Output STRICT JSON ONLY. No extra text.\n"
                            if structured is None:
                                structured = normalize_structured_schema({}, cut_num, pattern_id, unit_obj)
                                structured["consistency"]["notes"].append(f"structured_parse_failed: {last_err}")

                            prompt_desc = build_description_prompt(
                                cut_num=cut_num,
                                pattern_id=pattern_id,
                                unit_obj=unit_obj,
                                row_action_block=row_action_block,
                                row_dialogue_block=row_dialogue_block,
                                description_style_hint=style_hint,
                                structured_json=structured,
                            )
                            raw_desc = vl_generate_text(model, processor, img, prompt_desc, CFG["gen_desc"])
                            description = raw_desc.strip()

                            meta = {
                                "created_at": now_iso(),
                                "cut": cut_num,
                                "pattern": pattern_id,
                                "unit": unit_obj,
                                "stage1_path": s1_path,
                                "stage2_split_path": s2_path,
                                "stage2_variant": stage2_variant,
                                "model_id": CFG["model_id"],
                                "gen_struct": CFG["gen_struct"],
                                "gen_desc": CFG["gen_desc"],
                                "note": "SAME prompt template across patterns; only input image differs. Text blocks include ordered rows and raw/corrected.",
                            }

                            save_bundle(
                                out_dir,
                                image=img,
                                structured=structured,
                                description=description,
                                meta=meta,
                                raw_structured_text=raw_structured,
                                raw_description_text=raw_desc,
                                prompt_struct=prompt_struct,
                                prompt_desc=prompt_desc,
                            )

                # Pattern2: row concat image
                if "p2_row_concat" in CFG["patterns"]:
                    pattern_id = "p2_row_concat"
                    out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname, unit_folder)
                    if CFG["skip_if_exists"] and os.path.isfile(os.path.join(out_dir, "description.txt")):
                        continue

                    img = make_pattern2_image(stage1_row)
                    if img is None:
                        continue

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

                    raw_structured = ""
                    structured = None
                    last_err = None
                    for _ in range(int(CFG["max_structured_retries"]) + 1):
                        raw_structured = vl_generate_text(model, processor, img, prompt_struct, CFG["gen_struct"])
                        try:
                            parsed = parse_json_relaxed(raw_structured)
                            structured = normalize_structured_schema(parsed, cut_num, pattern_id, unit_obj)
                            break
                        except Exception as e:
                            last_err = str(e)
                            prompt_struct = prompt_struct + "\nREMINDER: Output STRICT JSON ONLY. No extra text.\n"
                    if structured is None:
                        structured = normalize_structured_schema({}, cut_num, pattern_id, unit_obj)
                        structured["consistency"]["notes"].append(f"structured_parse_failed: {last_err}")

                    prompt_desc = build_description_prompt(
                        cut_num=cut_num,
                        pattern_id=pattern_id,
                        unit_obj=unit_obj,
                        row_action_block=row_action_block,
                        row_dialogue_block=row_dialogue_block,
                        description_style_hint=style_hint,
                        structured_json=structured,
                    )
                    raw_desc = vl_generate_text(model, processor, img, prompt_desc, CFG["gen_desc"])
                    description = raw_desc.strip()

                    meta = {
                        "created_at": now_iso(),
                        "cut": cut_num,
                        "pattern": pattern_id,
                        "unit": unit_obj,
                        "stage1_path": s1_path,
                        "stage2_split_path": s2_path,
                        "stage2_variant": stage2_variant,
                        "model_id": CFG["model_id"],
                        "gen_struct": CFG["gen_struct"],
                        "gen_desc": CFG["gen_desc"],
                        "note": "SAME prompt template across patterns; only input image differs. Text blocks include ordered rows and raw/corrected.",
                    }

                    save_bundle(
                        out_dir,
                        image=img,
                        structured=structured,
                        description=description,
                        meta=meta,
                        raw_structured_text=raw_structured,
                        raw_description_text=raw_desc,
                        prompt_struct=prompt_struct,
                        prompt_desc=prompt_desc,
                    )

        # -------------------------
        # Pattern 3: cut-based
        # -------------------------
        if "p3_cut_concat" in CFG["patterns"]:
            pattern_id = "p3_cut_concat"
            out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname)
            if CFG["skip_if_exists"] and os.path.isfile(os.path.join(out_dir, "description.txt")):
                continue

            img = make_pattern3_image(stage1_cut)
            if img is None:
                continue

            unit_obj = {"cut": cut_num}

            # For cut-level: "current unit" row blocks can be the ordered whole cut blocks (still same template)
            row_action_block = cut_action_block
            row_dialogue_block = cut_dialogue_block

            style_hint = "2〜4文で、このカット全体（複数行の流れ）を踏まえて説明。絵が最優先。"

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

            raw_structured = ""
            structured = None
            last_err = None
            for _ in range(int(CFG["max_structured_retries"]) + 1):
                raw_structured = vl_generate_text(model, processor, img, prompt_struct, CFG["gen_struct"])
                try:
                    parsed = parse_json_relaxed(raw_structured)
                    structured = normalize_structured_schema(parsed, cut_num, pattern_id, unit_obj)
                    break
                except Exception as e:
                    last_err = str(e)
                    prompt_struct = prompt_struct + "\nREMINDER: Output STRICT JSON ONLY. No extra text.\n"
            if structured is None:
                structured = normalize_structured_schema({}, cut_num, pattern_id, unit_obj)
                structured["consistency"]["notes"].append(f"structured_parse_failed: {last_err}")

            prompt_desc = build_description_prompt(
                cut_num=cut_num,
                pattern_id=pattern_id,
                unit_obj=unit_obj,
                row_action_block=row_action_block,
                row_dialogue_block=row_dialogue_block,
                description_style_hint=style_hint,
                structured_json=structured,
            )
            raw_desc = vl_generate_text(model, processor, img, prompt_desc, CFG["gen_desc"])
            description = raw_desc.strip()

            meta = {
                "created_at": now_iso(),
                "cut": cut_num,
                "pattern": pattern_id,
                "unit": unit_obj,
                "stage1_path": s1_path,
                "stage2_split_path": s2_path,
                "stage2_variant": stage2_variant,
                "model_id": CFG["model_id"],
                "gen_struct": CFG["gen_struct"],
                "gen_desc": CFG["gen_desc"],
                "note": "SAME prompt template across patterns; only input image differs. Cut-level unit uses ordered cut text as row block.",
            }

            save_bundle(
                out_dir,
                image=img,
                structured=structured,
                description=description,
                meta=meta,
                raw_structured_text=raw_structured,
                raw_description_text=raw_desc,
                prompt_struct=prompt_struct,
                prompt_desc=prompt_desc,
            )


if __name__ == "__main__":
    main()

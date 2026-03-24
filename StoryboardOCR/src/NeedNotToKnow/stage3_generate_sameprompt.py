# stage3_generate_sameprompt.py
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
STAGE2_CUTS_DIR = f"{APP_ROOT}/outputs_stage2_clm/{EPISODE_ID}/cuts"
SYMBOL_LEXICON_PATH = f"{APP_ROOT}/data/storyboard_symbol_lexicon.json"
STAGE3_OUT_ROOT = f"{APP_ROOT}/outputs_stage3/{EPISODE_ID}"
# ==================================================


# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
CFG = {
    # which cuts to process
    "cut_select": "all",   # "range" or "range"
    "cut_start": 1,
    "cut_end": 10,

    # patterns to run
    "patterns": ["p1_row_picture", "p2_row_concat", "p3_cut_concat"],

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

    # RAG injection sizes (Stage2 extracted only)
    "max_script_evidence": 3,
    "max_terms_with_meaning": 25,

    # output saving
    "save_prompts": True,
    "save_raw_texts": True,

    # retries for structured JSON parse
    "max_structured_retries": 2,

    # output language
    "lang": "ja",
}
# ==================================================


# -------------------------
# File helpers
# -------------------------
CUT_FILE_RE = re.compile(r"cut(\d+)\.stage(\d)\.json$")
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

def list_cut_files(dir_path: str, stage: int) -> List[str]:
    out = []
    if not os.path.isdir(dir_path):
        return out
    for fn in os.listdir(dir_path):
        m = CUT_FILE_RE.match(fn)
        if not m:
            continue
        if int(m.group(2)) != stage:
            continue
        out.append(os.path.join(dir_path, fn))
    out.sort()
    return out

def parse_cut_num(path: str) -> Optional[int]:
    m = CUT_FILE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def cut_in_range(cut_num: Optional[int]) -> bool:
    if CFG["cut_select"] == "all":
        return True
    if cut_num is None:
        return False
    return int(CFG["cut_start"]) <= cut_num <= int(CFG["cut_end"])

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
# Lexicon / Stage2 helpers
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

def stage1_row_cell_text(row_obj: Dict[str, Any], col: str) -> str:
    cell = (row_obj.get("cols", {}) or {}).get(col, {}) or {}
    raw = cell.get("raw_text")
    return str(raw) if isinstance(raw, str) else ""

def stage2_cut_texts(stage2_obj: Dict[str, Any]) -> Tuple[str, str]:
    ocr_norm = stage2_obj.get("ocr_norm", {}) or {}
    return str(ocr_norm.get("action_memo", "") or ""), str(ocr_norm.get("dialogue", "") or "")

def get_stage2_external_knowledge(stage2: Dict[str, Any], term2meaning: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    rag = stage2.get("rag", {}) or {}
    script_evidence = (rag.get("script_evidence", []) or [])[: int(CFG["max_script_evidence"])]
    used_terms = rag.get("used_terms", []) or []

    terms_with_meaning = []
    for t in used_terms:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        terms_with_meaning.append({"term": tt, "meaning": term2meaning.get(tt, "")})

    terms_with_meaning = terms_with_meaning[: int(CFG["max_terms_with_meaning"])]
    return script_evidence, terms_with_meaning


# -------------------------
# Stage1/Stage2 matching (robust by cut number)
# -------------------------
def build_cutnum_to_path_map(cuts_dir: str, stage: int) -> Dict[int, str]:
    """
    Map cut_num -> file_path by filename pattern.
    If duplicates exist, pick the newest file (mtime).
    """
    mp: Dict[int, str] = {}
    mt: Dict[int, float] = {}
    for p in list_cut_files(cuts_dir, stage=stage):
        c = parse_cut_num(p)
        if c is None:
            continue
        t = os.path.getmtime(p)
        if (c not in mp) or (t > mt.get(c, -1.0)):
            mp[c] = p
            mt[c] = t
    return mp


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

    # pad token fallback
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
    Try strict JSON first, then repair common issues,
    then fallback to ast.literal_eval for python-ish dict outputs.
    """
    t = _strip_code_fences(text)
    t = _extract_brace_block(t)
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    t2 = _remove_trailing_commas(t)

    try:
        return json.loads(t2)
    except Exception:
        pass

    # fallback literal_eval (safe)
    t3 = t2
    t3 = re.sub(r"\bnull\b", "None", t3, flags=re.IGNORECASE)
    t3 = re.sub(r"\btrue\b", "True", t3, flags=re.IGNORECASE)
    t3 = re.sub(r"\bfalse\b", "False", t3, flags=re.IGNORECASE)

    obj = ast.literal_eval(t3)
    if isinstance(obj, dict):
        return obj
    raise json.JSONDecodeError("Failed to parse JSON (relaxed)", text, 0)

def normalize_structured_schema(obj: Dict[str, Any], cut_num: int, pattern_id: str, unit_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure keys exist for downstream processing.
    """
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
    ev.setdefault("used_script_chunk_ids", [])
    ev.setdefault("used_terms_with_meaning", [])
    ev.setdefault("image_observations", [])
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
def build_structured_prompt(
    *,
    cut_num: int,
    pattern_id: str,
    unit_obj: Dict[str, Any],
    row_action: str,
    row_dialogue: str,
    cut_action_norm: str,
    cut_dialogue_norm: str,
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
        "1) The IMAGE is the primary evidence.\n"
        "2) OCR text and retrieved script are only hints.\n"
        "If the text conflicts with the image, prioritize the image and note the conflict.\n\n"
        "Rules:\n"
        "- Output must be STRICT JSON. Use double quotes for all strings.\n"
        "- Do NOT output markdown code fences.\n"
        "- Do NOT invent details that are not supported by the image.\n"
        "- If unknown, use null or empty list.\n"
        "- You MUST include at least 3 short image observations in evidence.image_observations.\n"
        "- Camera movement must be included ONLY if it is visible in the image or explicitly written in OCR.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        f"=== ROW OCR (Stage1, hint) ===\n[action_memo_row]\n{row_action}\n\n[dialogue_row]\n{row_dialogue}\n\n"
        f"=== CUT OCR (Stage2, hint) ===\n[action_memo_cut]\n{cut_action_norm}\n\n[dialogue_cut]\n{cut_dialogue_norm}\n\n"
        f"=== SCRIPT (retrieved, hint) ===\n{script_block}\n\n"
        f"=== TERMS (meaning attached, hint) ===\n{terms_block}\n\n"
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
    row_action: str,
    row_dialogue: str,
    structured_json: Dict[str, Any],
) -> str:
    prompt = (
        "You write a short Japanese scene description for an anime storyboard.\n"
        "You are given an input image, OCR text, and a structured JSON analysis.\n\n"
        "CRITICAL PRIORITY:\n"
        "1) The IMAGE is the primary evidence.\n"
        "2) The structured JSON is only a checklist to avoid missing points.\n"
        "If the JSON conflicts with the image, prioritize the image and ignore the conflicting JSON parts.\n\n"
        "Rules:\n"
        "- Do NOT invent details.\n"
        "- 2-4 sentences in Japanese.\n"
        "- Mention camera movement only if it is visible in the image or explicitly written in OCR.\n"
        "- Include at least one concrete visual fact from the image.\n"
        "- Dialogue should be included only if supported by OCR.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        f"=== ROW OCR (hint) ===\n[action_memo_row]\n{row_action}\n\n[dialogue_row]\n{row_dialogue}\n\n"
        f"=== STRUCTURED JSON (checklist, not primary) ===\n{json.dumps(structured_json, ensure_ascii=False)}\n\n"
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

def make_pattern1_image(row_obj: Dict[str, Any]) -> Optional[Image.Image]:
    cols = row_obj.get("cols", {}) or {}
    pic = cols.get("picture", {}) or {}
    p = pic.get("image")
    if not isinstance(p, str) or not p:
        return None
    p = resolve_path(p)
    if not os.path.isfile(p):
        return None
    im = open_rgb(p)
    return resize_to_max_side(im, int(CFG["max_image_side"]))

def make_pattern2_image(row_obj: Dict[str, Any]) -> Optional[Image.Image]:
    paths = build_row_images_from_stage1_row(row_obj)
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
    for row_obj in stage1_cut.get("rows", []) or []:
        im = make_pattern2_image(row_obj)
        if im is not None:
            row_imgs.append(im)
    if not row_imgs:
        return None
    out = vconcat(row_imgs, int(CFG["cut_concat_pad"]), CFG["cut_concat_bg"])
    return resize_to_max_side(out, int(CFG["max_image_side"]))


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

    # match stage1/stage2 by cut number
    s2_map = build_cutnum_to_path_map(STAGE2_CUTS_DIR, stage=2)
    s1_map = build_cutnum_to_path_map(STAGE1_CUTS_DIR, stage=1)

    cut_nums = sorted([c for c in s2_map.keys() if cut_in_range(c)])
    if not cut_nums:
        raise FileNotFoundError(f"No stage2 cut files found in range: {STAGE2_CUTS_DIR}")

    for cut_num in tqdm(cut_nums, desc="Stage3: Cuts"):
        s2_path = s2_map.get(cut_num)
        s1_path = s1_map.get(cut_num)
        if not s2_path or not os.path.isfile(s2_path):
            continue
        if not s1_path or not os.path.isfile(s1_path):
            continue

        stage2 = read_json(s2_path)
        stage1 = read_json(s1_path)

        cut_action_norm, cut_dialogue_norm = stage2_cut_texts(stage2)
        script_evidence, terms_with_meaning = get_stage2_external_knowledge(stage2, term2meaning)

        cut_dirname = f"cut{int(cut_num):04d}"
        rows = stage1.get("rows", []) or []

        # -------------------------
        # Pattern 1 / 2: row-based
        # -------------------------
        if ("p1_row_picture" in CFG["patterns"]) or ("p2_row_concat" in CFG["patterns"]):
            for row_obj in tqdm(rows, desc=f"Cut {cut_num}: Rows", leave=False):
                try:
                    page = int(row_obj.get("page", -1))
                    row = int(row_obj.get("row", -1))
                except Exception:
                    continue
                if page < 0 or row < 0:
                    continue

                unit_folder = f"page{page:02d}row{row:02d}"
                unit_obj = {"page": page, "row": row}

                # SAME prompt template; values filled from row/cut
                row_action = stage1_row_cell_text(row_obj, "action_memo")
                row_dialogue = stage1_row_cell_text(row_obj, "dialogue")

                # Pattern1: picture image only
                if "p1_row_picture" in CFG["patterns"]:
                    pattern_id = "p1_row_picture"
                    out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname, unit_folder)
                    if CFG["skip_if_exists"] and os.path.isfile(os.path.join(out_dir, "description.txt")):
                        pass
                    else:
                        img = make_pattern1_image(row_obj)
                        if img is not None:
                            prompt_struct = build_structured_prompt(
                                cut_num=cut_num,
                                pattern_id=pattern_id,
                                unit_obj=unit_obj,
                                row_action=row_action,
                                row_dialogue=row_dialogue,
                                cut_action_norm=cut_action_norm,
                                cut_dialogue_norm=cut_dialogue_norm,
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
                                row_action=row_action,
                                row_dialogue=row_dialogue,
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
                                "stage2_path": s2_path,
                                "model_id": CFG["model_id"],
                                "gen_struct": CFG["gen_struct"],
                                "gen_desc": CFG["gen_desc"],
                                "note": "SAME prompt template across patterns; only input image differs.",
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

                    img = make_pattern2_image(row_obj)
                    if img is None:
                        continue

                    prompt_struct = build_structured_prompt(
                        cut_num=cut_num,
                        pattern_id=pattern_id,
                        unit_obj=unit_obj,
                        row_action=row_action,
                        row_dialogue=row_dialogue,
                        cut_action_norm=cut_action_norm,
                        cut_dialogue_norm=cut_dialogue_norm,
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
                        row_action=row_action,
                        row_dialogue=row_dialogue,
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
                        "stage2_path": s2_path,
                        "model_id": CFG["model_id"],
                        "gen_struct": CFG["gen_struct"],
                        "gen_desc": CFG["gen_desc"],
                        "note": "SAME prompt template across patterns; only input image differs.",
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

            img = make_pattern3_image(stage1)
            if img is None:
                continue

            unit_obj = {"cut": cut_num}

            # For cut-level unit: we still keep SAME prompt template.
            # Row OCR section is filled with concatenated Stage1 row texts (optional but consistent and transparent).
            action_lines = []
            dialog_lines = []
            for ro in rows:
                p = ro.get("page")
                r = ro.get("row")
                a = stage1_row_cell_text(ro, "action_memo")
                d = stage1_row_cell_text(ro, "dialogue")
                if isinstance(p, int) and isinstance(r, int):
                    tag = f"[page={p} row={r}]"
                else:
                    tag = "[row]"
                if a.strip():
                    action_lines.append(f"{tag}\n{a}")
                if d.strip():
                    dialog_lines.append(f"{tag}\n{d}")

            row_action_concat = "\n\n".join(action_lines).strip()
            row_dialogue_concat = "\n\n".join(dialog_lines).strip()

            prompt_struct = build_structured_prompt(
                cut_num=cut_num,
                pattern_id=pattern_id,
                unit_obj=unit_obj,
                row_action=row_action_concat,
                row_dialogue=row_dialogue_concat,
                cut_action_norm=cut_action_norm,
                cut_dialogue_norm=cut_dialogue_norm,
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
                row_action=row_action_concat,
                row_dialogue=row_dialogue_concat,
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
                "stage2_path": s2_path,
                "model_id": CFG["model_id"],
                "gen_struct": CFG["gen_struct"],
                "gen_desc": CFG["gen_desc"],
                "note": "SAME prompt template across patterns; only input image differs.",
                "row_ocr_concat": {
                    "action_memo_chars": len(row_action_concat),
                    "dialogue_chars": len(row_dialogue_concat),
                },
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

# stage3_generate.py

import os
import re
import json
from tqdm import tqdm
from dataclasses import dataclass
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

# Stage1/Stage2 inputs
STAGE1_CUTS_DIR = f"{APP_ROOT}/outputs/{EPISODE_ID}/cuts"
STAGE2_CUTS_DIR = f"{APP_ROOT}/outputs_stage2/{EPISODE_ID}/cuts"

# Symbol lexicon (id, word[], category, 意味)
SYMBOL_LEXICON_PATH = f"{APP_ROOT}/data/storyboard_symbol_lexicon.json"

# Output root
STAGE3_OUT_ROOT = f"{APP_ROOT}/outputs_stage3/{EPISODE_ID}"
# ==================================================

# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
CFG = {
    # which cuts to process
    "cut_select": "all",  # "all" or "range"
    "cut_start": 1,
    "cut_end": 10,

    # pattern ON/OFF (edit this list)
    # p1_row_picture: row picture only
    # p2_row_concat : row concat (cut|picture|action|dialogue|time)
    # p3_cut_concat : cut concat (stack row-concats)
    "patterns": ["p1_row_picture", "p2_row_concat", "p3_cut_concat"],

    # skip if output already exists
    "skip_if_exists": True,

    # concat layout
    "row_concat_order": ["cut", "picture", "action_memo", "dialogue", "time"],  # left->right
    "row_concat_pad": 10,
    "row_concat_bg": (255, 255, 255),

    "cut_concat_pad": 10,
    "cut_concat_bg": (255, 255, 255),

    # model
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,
    "gen_struct": {"max_new_tokens": 2048, "do_sample": False, "num_beams": 1},
    "gen_desc":   {"max_new_tokens": 512, "do_sample": False, "num_beams": 1},

    # RAG injection sizes (Stage2 extracted only)
    "max_script_evidence": 3,     # from stage2["rag"]["script_evidence"]
    "max_terms_with_meaning": 25, # from stage2["rag"]["used_terms"]

    # output language
    "lang": "ja",
}
# ==================================================

# -------------------------
# Helpers
# -------------------------
CUT_FILE_RE = re.compile(r"cut(\d+)\.stage(\d)\.json$")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    return CFG["cut_start"] <= cut_num <= CFG["cut_end"]

def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

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
    script_evidence = (rag.get("script_evidence", []) or [])[:CFG["max_script_evidence"]]
    used_terms = rag.get("used_terms", []) or []

    terms_with_meaning = []
    for t in used_terms:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        terms_with_meaning.append({"term": tt, "meaning": term2meaning.get(tt, "")})

    terms_with_meaning = terms_with_meaning[:CFG["max_terms_with_meaning"]]
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
    return model, processor

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
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out_ids = model.generate(**inputs, **gen_cfg)
    trimmed = out_ids[0][len(inputs["input_ids"][0]):]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

def parse_json_from_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        raise


# -------------------------
# Prompts (minimal but strong schema)
# -------------------------
def build_structured_prompt(
    *,
    cut_num: int,
    pattern_id: str,
    unit_obj: Dict[str, Any],
    # row-specific (Stage1) primary
    row_action: str,
    row_dialogue: str,
    # cut-level (Stage2) secondary
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
        "You analyze an anime storyboard and output JSON only.\n"
        "You are given an input image and text/context.\n\n"
        "Rules:\n"
        "- Do NOT invent details not supported by the image or provided text.\n"
        "- If unknown, use null or empty list.\n"
        "- Prefer ROW text as the primary evidence for row-based patterns.\n"
        "- Use CUT-level text only as background context.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        f"=== ROW OCR (primary) ===\n[action_memo_row]\n{row_action}\n\n[dialogue_row]\n{row_dialogue}\n\n"
        f"=== CUT OCR (secondary, Stage2) ===\n[action_memo_cut]\n{cut_action_norm}\n\n[dialogue_cut]\n{cut_dialogue_norm}\n\n"
        f"=== SCRIPT (retrieved) ===\n{script_block}\n\n"
        f"=== TERMS (meaning attached) ===\n{terms_block}\n\n"
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
        "You are given an input image, ROW OCR text, and a structured JSON analysis.\n\n"
        "Rules:\n"
        "- Use the structured JSON as the primary guide.\n"
        "- Do NOT invent details.\n"
        "- 2-4 sentences in Japanese.\n"
        "- Mention camera movement if present.\n\n"
        f"=== META ===\ncut={cut_num}\npattern={pattern_id}\nunit={json.dumps(unit_obj, ensure_ascii=False)}\n\n"
        f"=== ROW OCR ===\n[action_memo_row]\n{row_action}\n\n[dialogue_row]\n{row_dialogue}\n\n"
        f"=== STRUCTURED JSON ===\n{json.dumps(structured_json, ensure_ascii=False)}\n\n"
        "=== OUTPUT ===\nReturn ONLY the description text.\n"
    )
    return prompt


# -------------------------
# Image building for patterns
# -------------------------
def build_row_images_from_stage1_row(row_obj: Dict[str, Any]) -> Dict[str, str]:
    cols = row_obj.get("cols", {}) or {}
    img_paths = {}
    for k in CFG["row_concat_order"]:
        cell = cols.get(k, {}) or {}
        p = cell.get("image")
        if isinstance(p, str) and p:
            img_paths[k] = os.path.join(APP_ROOT, p) if not os.path.isabs(p) else p
        else:
            img_paths[k] = ""
    return img_paths

def make_pattern1_image(row_obj: Dict[str, Any]) -> Optional[Image.Image]:
    cols = row_obj.get("cols", {}) or {}
    pic = cols.get("picture", {}) or {}
    p = pic.get("image")
    if not isinstance(p, str) or not p:
        return None
    p = os.path.join(APP_ROOT, p) if not os.path.isabs(p) else p
    if not os.path.isfile(p):
        return None
    return open_rgb(p)

def make_pattern2_image(row_obj: Dict[str, Any]) -> Optional[Image.Image]:
    paths = build_row_images_from_stage1_row(row_obj)
    ims = []
    for k in CFG["row_concat_order"]:
        p = paths.get(k, "")
        if p and os.path.isfile(p):
            ims.append(open_rgb(p))
        else:
            ims.append(Image.new("RGB", (200, 200), CFG["row_concat_bg"]))
    return hconcat(ims, CFG["row_concat_pad"], CFG["row_concat_bg"])

def make_pattern3_image(stage1_cut: Dict[str, Any]) -> Optional[Image.Image]:
    row_imgs = []
    for row_obj in stage1_cut.get("rows", []):
        im = make_pattern2_image(row_obj)
        if im is not None:
            row_imgs.append(im)
    if not row_imgs:
        return None
    return vconcat(row_imgs, CFG["cut_concat_pad"], CFG["cut_concat_bg"])


# -------------------------
# Save
# -------------------------
def save_bundle(base_dir: str, image: Image.Image, structured: Dict[str, Any], description: str):
    ensure_dir(base_dir)
    img_path = os.path.join(base_dir, "input.png")
    json_path = os.path.join(base_dir, "structured.json")
    txt_path = os.path.join(base_dir, "description.txt")

    image.save(img_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(description.strip() + "\n")


# -------------------------
# Main
# -------------------------
def main():
    ensure_dir(STAGE3_OUT_ROOT)

    term2meaning = load_symbol_lexicon(SYMBOL_LEXICON_PATH)
    model, processor = load_model()

    stage2_files = list_cut_files(STAGE2_CUTS_DIR, stage=2)
    stage2_files = [p for p in stage2_files if cut_in_range(parse_cut_num(p))]

    if not stage2_files:
        raise FileNotFoundError(f"No stage2 cut files found in range: {STAGE2_CUTS_DIR}")

    for s2_path in tqdm(stage2_files, desc="Stage3: Cuts"):
        cut_num = parse_cut_num(s2_path)
        if cut_num is None:
            continue

        stage2 = read_json(s2_path)

        # corresponding stage1 path (same cut id)
        s1_name = os.path.basename(s2_path).replace(".stage2.json", ".stage1.json")
        s1_path = os.path.join(STAGE1_CUTS_DIR, s1_name)
        if not os.path.isfile(s1_path):
            continue
        stage1 = read_json(s1_path)

        cut_action_norm, cut_dialogue_norm = stage2_cut_texts(stage2)
        script_evidence, terms_with_meaning = get_stage2_external_knowledge(stage2, term2meaning)

        cut_dirname = f"cut{int(cut_num):04d}"

        rows = stage1.get("rows", [])

        # -------------------------
        # Pattern 1 / 2: row-based
        # -------------------------
        if ("p1_row_picture" in CFG["patterns"]) or ("p2_row_concat" in CFG["patterns"]):
            for row_obj in tqdm(rows, desc=f"Cut {cut_num}: Rows", leave=False):
                page = int(row_obj.get("page", -1))
                row = int(row_obj.get("row", -1))
                if page < 0 or row < 0:
                    continue

                unit_folder = f"page{page:02d}row{row:02d}"
                unit_obj = {"page": page, "row": row}

                # row-specific texts (Stage1 primary)
                row_action = stage1_row_cell_text(row_obj, "action_memo")
                row_dialogue = stage1_row_cell_text(row_obj, "dialogue")

                # Pattern1
                if "p1_row_picture" in CFG["patterns"]:
                    pattern_id = "p1_row_picture"
                    out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname, unit_folder)
                    if CFG["skip_if_exists"] and os.path.isfile(os.path.join(out_dir, "description.txt")):
                        continue

                    img = make_pattern1_image(row_obj)
                    if img is None:
                        continue

                    # Step A (structured)
                    print(f"[Stage3] cut={cut_num} {unit_folder} {pattern_id} structured")
                    sp = build_structured_prompt(
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
                    structured_text = vl_generate_text(model, processor, img, sp, CFG["gen_struct"])
                    structured = parse_json_from_text(structured_text)

                    # Step B (description)
                    print(f"[Stage3] cut={cut_num} {unit_folder} {pattern_id} description")
                    dp = build_description_prompt(
                        cut_num=cut_num,
                        pattern_id=pattern_id,
                        unit_obj=unit_obj,
                        row_action=row_action,
                        row_dialogue=row_dialogue,
                        structured_json=structured,
                    )
                    description = vl_generate_text(model, processor, img, dp, CFG["gen_desc"])

                    save_bundle(out_dir, img, structured, description)

                # Pattern2
                if "p2_row_concat" in CFG["patterns"]:
                    pattern_id = "p2_row_concat"
                    out_dir = os.path.join(STAGE3_OUT_ROOT, pattern_id, cut_dirname, unit_folder)
                    if CFG["skip_if_exists"] and os.path.isfile(os.path.join(out_dir, "description.txt")):
                        continue

                    img = make_pattern2_image(row_obj)
                    if img is None:
                        continue

                    print(f"[Stage3] cut={cut_num} {unit_folder} {pattern_id} structured")
                    sp = build_structured_prompt(
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
                    structured_text = vl_generate_text(model, processor, img, sp, CFG["gen_struct"])
                    structured = parse_json_from_text(structured_text)

                    print(f"[Stage3] cut={cut_num} {unit_folder} {pattern_id} description")
                    dp = build_description_prompt(
                        cut_num=cut_num,
                        pattern_id=pattern_id,
                        unit_obj=unit_obj,
                        row_action=row_action,
                        row_dialogue=row_dialogue,
                        structured_json=structured,
                    )
                    description = vl_generate_text(model, processor, img, dp, CFG["gen_desc"])

                    save_bundle(out_dir, img, structured, description)

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

            # For cut-level, primary texts = Stage2 cut-level
            row_action = cut_action_norm
            row_dialogue = cut_dialogue_norm

            print(f"[Stage3] cut={cut_num} {pattern_id} structured")
            sp = build_structured_prompt(
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
            structured_text = vl_generate_text(model, processor, img, sp, CFG["gen_struct"])
            structured = parse_json_from_text(structured_text)

            print(f"[Stage3] cut={cut_num} {pattern_id} description")
            dp = build_description_prompt(
                cut_num=cut_num,
                pattern_id=pattern_id,
                unit_obj=unit_obj,
                row_action=row_action,
                row_dialogue=row_dialogue,
                structured_json=structured,
            )
            description = vl_generate_text(model, processor, img, dp, CFG["gen_desc"])

            save_bundle(out_dir, img, structured, description)


if __name__ == "__main__":
    main()

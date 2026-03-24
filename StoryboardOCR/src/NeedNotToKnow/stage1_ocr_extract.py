# stage1_ocr_extract.py

import os
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# Optional (for LoRA adapter)
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # Input
    "episode_id": "free1",
    "episode_dir": "data/free/free1",
    "script_phase": "script_phase1",

    # Output base dir
    "out_dir": "outputs/free",

    # Page selection
    "page_select": "all",  # "all" or "range"
    "start_page": 2,
    "end_page": 55,

    # Usually 5 rows per storyboard page in your preprocessing
    "rows_per_page": 5,

    # OCR columns
    "ocr_columns": ["cut", "picture", "action_memo", "dialogue", "time"],

    # Base Model
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,

    # Generation
    "gen": {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
    },

    # LoRA adapter option (fine-tuned OCR model)
    # If you trained LoRA, put adapter directory here.
    "use_lora_adapter": False,
    "lora_adapter_path": "outputs_ft/stage1_ocr_lora_full",  # example

    # If True, skip writing a cut file when it already exists
    "skip_if_exists": False,

    # If picture is not OCRed, we still store picture image path for Stage2
    "always_store_picture_path": True,

    # storyboard symbol lexicon (terms + meaning)
    "symbol_lexicon_path": "data/storyboard_symbol_lexicon.json",

    # character lexicon (Stage0 output)
    # e.g. outputs/script_phase1/episode01/lexicon_entities.json
    "lexicon_path": "lexicon_entities.json",
}
# =========================


# -------------------------
# Column-specific term-category selection
# -------------------------
COLUMN_TERM_CATEGORIES = {
    # cut column: mostly digits/time/continued; keep minimal categories
    "cut": ["structure", "editing", "timing"],

    # picture: mostly drawings; if text exists it's often camera/framing arrows and short notes
    "picture": ["camera_movement", "camera_angle", "framing", "focus", "timing", "editing"],

    # action memo: production/camera/timing/effects heavy
    "action_memo": [
        "camera_movement", "camera_angle", "framing", "focus",
        "timing", "effect", "animation", "editing", "structure", "production", "expression", "sound", "dialogue"
    ],

    # dialogue: speaker, quotes, SE/MA/BGM, off/on, monologue markers
    "dialogue": ["dialogue", "sound", "editing", "structure"],

    # time: timing expressions
    "time": ["timing"],
}


# -------------------------
# Prompt templates (per column)
# -------------------------
BASE_RULES = (
    "Rules:\n"
    "- Keep original line breaks.\n"
    "- Do NOT correct typos.\n"
    "- Do NOT translate.\n"
    "- If unreadable, output '□'.\n"
    "- Return ONLY the transcribed text. No explanations.\n"
)

HANDWRITING_NOTE = (
    "Notes:\n"
    "- This storyboard contains BOTH handwritten text and computer-typed text.\n"
    "- Handwritten text may include arrows and symbols.\n"
    "- Arrows are important. Always transcribe arrows if present.\n\n"
)

COLUMN_BASE_PROMPTS = {
    "cut": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: CUT.\n"
        "Task: Read the CUT NUMBER if present.\n"
        "It may also include a time like mm:ss.\n"
        "Handwritten digits may appear.\n"
        "If nothing readable, output '□'.\n\n"
        f"{BASE_RULES}\n"
        f"{HANDWRITING_NOTE}"
        "Extra constraints for CUT column:\n"
        "- Prefer digits and short markers.\n"
        "- If both a cut number and a time exist, keep both on separate lines as seen.\n"
        "- Do NOT guess missing digits.\n\n"
    ),

    "picture": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: PICTURE (mostly drawings).\n"
        "Task: Transcribe ANY text that appears inside the picture area.\n"
        "Most of the region is drawings, but sometimes handwritten terms exist (PAN, arrows, etc.).\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, brackets, punctuation).\n"
        "If there is no text, output '□'.\n\n"
        f"{BASE_RULES}\n"
        f"{HANDWRITING_NOTE}"
        "Extra constraints for PICTURE column:\n"
        "- Only transcribe visible text/symbols; ignore the drawings.\n"
        "- Handwritten arrows and short terms are common; keep them exactly.\n"
        "- If arrows are attached to a term (e.g., →PAN), keep them together as seen.\n\n"
    ),

    "action_memo": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: ACTION MEMO.\n"
        "Task: Transcribe all handwritten/typed notes about action, camera, timing, and effects.\n"
        "This column often contains many production terms (PAN/FIX/ヨリ/トメ/etc.).\n\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, brackets, punctuation).\n"
        f"{BASE_RULES}\n"
        f"{HANDWRITING_NOTE}"
        "Extra constraints for ACTION MEMO column:\n"
        "- Preserve production terms exactly.\n"
        "- Keep parentheses and arrows exactly.\n"
        "- Do NOT normalize or rewrite.\n\n"
    ),

    "dialogue": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: DIALOGUE.\n"
        "Task: Transcribe dialogue lines and related audio notes.\n"
        "This column may include speaker names, quoted lines 「」, monologue markers (M/Ｍ), and sound notes (SE/MA/BGM/ガヤ/off/on).\n\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, brackets, punctuation).\n"
        f"{BASE_RULES}\n"
        f"{HANDWRITING_NOTE}"
        "Extra constraints for DIALOGUE column:\n"
        "- Preserve speaker names and symbols (「」, （）).\n"
        "- Preserve audio tags like SE:, MA:, BGM:, ガヤ, off/on exactly.\n"
        "- Do NOT invent punctuation.\n\n"
    ),

    "time": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: TIME.\n"
        "Task: Transcribe time information if present.\n"
        "Common formats: mm:ss, (m+n), frame counts, or short numeric timing notes.\n"
        "If there is no readable timing, output '□'.\n\n"
        f"{BASE_RULES}\n"
        f"{HANDWRITING_NOTE}"
        "Extra constraints for TIME column:\n"
        "- Prefer numeric/time expressions.\n"
        "- Keep exactly as written.\n\n"
    ),
}


# -------------------------
# Lexicon loaders
# -------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lexicon_characters(path_json: str) -> List[str]:
    """
    Stage0 lexicon_entities.json から登場人物名を抽出（canonical）
    """
    if not os.path.isfile(path_json):
        return []
    obj = read_json(path_json)
    chars = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            name = e.get("canonical")
            if isinstance(name, str) and name.strip():
                chars.append(name.strip())
    chars.sort(key=lambda x: (-len(x), x))
    return chars


def load_symbol_terms_by_category(path_json: str) -> Dict[str, List[str]]:
    """
    storyboard_symbol_lexicon.json から category -> terms の形で抽出する。
    JSON形式（配列）:
      [{"id":..., "word":[...], "category":..., "意味":...}, ...]
    """
    if not os.path.isfile(path_json):
        return {}

    data = read_json(path_json)
    cat2terms: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}

    if isinstance(data, list):
        for item in data:
            cat = item.get("category", "unknown")
            if not isinstance(cat, str) or not cat:
                cat = "unknown"
            words = item.get("word", [])
            if not isinstance(words, list):
                continue

            if cat not in cat2terms:
                cat2terms[cat] = []
                seen[cat] = set()

            for w in words:
                if not isinstance(w, str):
                    continue
                t = w.strip()
                if not t:
                    continue
                if t not in seen[cat]:
                    seen[cat].add(t)
                    cat2terms[cat].append(t)

    # sort each category (longer first)
    for cat, terms in cat2terms.items():
        terms.sort(key=lambda x: (-len(x), x))
    return cat2terms


# -------------------------
# Prompt builder (column-specific, with meta blocks)
# -------------------------
def make_terms_block_for_column(symbol_terms_by_cat: Dict[str, List[str]], col: str) -> str:
    cats = COLUMN_TERM_CATEGORIES.get(col, [])
    if not cats:
        return ""

    lines = ["=== STORYBOARD TERMS (terms only) ==="]
    for cat in cats:
        terms = symbol_terms_by_cat.get(cat, [])
        if not terms:
            continue
        lines.append(f"[{cat}]")
        lines.extend(terms[:200])
    return "\n".join(lines) + "\n\n"


def make_character_block_for_column(lexicon_characters: List[str], col: str) -> str:
    # Character names help mostly in dialogue/action_memo; avoid adding in cut/time/picture by default.
    if col not in {"dialogue", "action_memo"}:
        return ""
    if not lexicon_characters:
        return ""
    return "=== CHARACTER NAMES (terms only) ===\n" + "\n".join(lexicon_characters[:200]) + "\n\n"


def build_column_prompt(col: str, lexicon_characters: List[str], symbol_terms_by_cat: Dict[str, List[str]]) -> str:
    base = COLUMN_BASE_PROMPTS.get(col, "You are an OCR engine.\n" + BASE_RULES + "\n")
    char_block = make_character_block_for_column(lexicon_characters, col)
    terms_block = make_terms_block_for_column(symbol_terms_by_cat, col)
    return base + char_block + terms_block + "Return ONLY the transcribed text.\n"


# -------------------------
# File patterns / helpers
# -------------------------
PAGE_RE = re.compile(r"^page(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
ROW_IMG_NAME = "page{page}_row{row}.png"

TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
CONTINUE_WORD_RE = re.compile(r"(続|つづく|継続|続き|つづき|CONT|CONTINUED)", re.IGNORECASE)
CUT_WORD_RE = re.compile(r"(CUT|ＣＵＴ|C\s*U\s*T)", re.IGNORECASE)

DIGIT_CONFUSION = {
    "0": ["9", "8"],
    "9": ["0"],
    "8": ["0"],
    "1": ["7"],
    "7": ["1"],
    "2": ["3"],
    "3": ["2"],
    "5": ["6"],
    "6": ["5"],
}


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def episode_id_from_dir(episode_dir: str) -> str:
    return os.path.basename(os.path.normpath(episode_dir))


def list_pages(episode_dir: str) -> List[int]:
    pages_dir = os.path.join(episode_dir, "pages")
    if not os.path.isdir(pages_dir):
        raise RuntimeError(f"pages/ not found: {pages_dir}")

    nums = []
    for name in os.listdir(pages_dir):
        m = PAGE_RE.match(name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def page_image_path(episode_dir: str, page: int) -> str:
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(episode_dir, "pages", f"page{page}.{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(episode_dir, "pages", f"page{page}.png")


def crop_path(episode_dir: str, col: str, page: int, row: int) -> str:
    return os.path.join(episode_dir, col, ROW_IMG_NAME.format(page=page, row=row))


def load_model_and_processor():
    kwargs = {
        "device_map": CFG["device_map"],
        "torch_dtype": CFG["torch_dtype"],
    }
    if CFG["flash_attn2"]:
        kwargs["attn_implementation"] = "flash_attention_2"

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(CFG["model_id"], **kwargs)

    # Optional: load LoRA adapter (fine-tuned)
    if CFG["use_lora_adapter"]:
        if PeftModel is None:
            raise RuntimeError("peft is not available, but use_lora_adapter=True. Install peft.")
        adapter_path = CFG["lora_adapter_path"]
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")
        base_model = PeftModel.from_pretrained(base_model, adapter_path)

    processor = AutoProcessor.from_pretrained(CFG["model_id"])
    base_model.eval()
    return base_model, processor


@torch.no_grad()
def ocr_one_image(model, processor, image: Image.Image, prompt_text: str) -> Dict[str, Any]:
    """
    OCR as plain text (no JSON).
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
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

    out_ids = model.generate(**inputs, **CFG["gen"])
    trimmed = out_ids[0][len(inputs["input_ids"][0]):]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()

    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    return {"raw_text": text, "lines": lines, "script": "unknown", "notes": []}


def extract_cut_candidates(raw_text: str, lines: List[str]) -> Dict[str, Any]:
    t = " ".join([raw_text] + (lines or []))
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    has_time = bool(TIME_TOKEN_RE.search(t))
    has_continue_word = bool(CONTINUE_WORD_RE.search(t))
    has_cut_word = bool(CUT_WORD_RE.search(t))

    t_wo_time = TIME_TOKEN_RE.sub(" ", t)

    m = re.search(r"(CUT|ＣＵＴ|C\s*U\s*T)\s*([0-9]{1,4})", t_wo_time, flags=re.IGNORECASE)
    explicit_cut_str = m.group(2) if m else None

    num_strs = re.findall(r"\b\d{1,4}\b", t_wo_time)

    nums: List[int] = []
    seen = set()
    for s in num_strs:
        n = int(s)
        if n not in seen:
            seen.add(n)
            nums.append(n)

    return {
        "normalized_text": t,
        "has_time": has_time,
        "has_continue_word": has_continue_word,
        "has_cut_word": has_cut_word,
        "explicit_cut_str": explicit_cut_str,
        "num_strs": num_strs,
        "numbers": nums,
    }


@dataclass
class CutSpan:
    cut_num: Optional[int]
    cut_num_str: Optional[str]
    start: Tuple[int, int]
    end: Tuple[int, int]
    anchor: Dict[str, Any]


def decide_cut_transition(
    *,
    previous_cut_num: Optional[int],
    candidates: Dict[str, Any],
    page: int,
    row: int,
) -> Tuple[Optional[int], Optional[str], bool, List[str], str]:
    reasons: List[str] = []
    nums: List[int] = candidates["numbers"]
    num_strs: List[str] = candidates["num_strs"]
    explicit_cut_str: Optional[str] = candidates["explicit_cut_str"]

    if not nums:
        return previous_cut_num, None, False, ["no_numeric_token"], "none"

    if previous_cut_num is not None and previous_cut_num in nums:
        reasons.append("same_as_previous_cut")
        if candidates["has_continue_word"]:
            reasons.append("explicit_continue_word")
        if row == 1:
            reasons.append("page_start_row")

        best_str = None
        for s in num_strs:
            if int(s) == previous_cut_num:
                if best_str is None or len(s) > len(best_str):
                    best_str = s
        if best_str:
            reasons.append("preserve_zero_padded_str")
        return previous_cut_num, best_str, False, reasons, "high" if candidates["has_continue_word"] else "medium"

    new_cut = nums[0]
    reasons.append("new_cut_number")

    best_str = None
    if explicit_cut_str and int(explicit_cut_str) == new_cut:
        best_str = explicit_cut_str
        reasons.append("explicit_cut_form")
    else:
        for s in num_strs:
            if int(s) == new_cut:
                if best_str is None or len(s) > len(best_str):
                    best_str = s
        if best_str and best_str.startswith("0"):
            reasons.append("leading_zeros_detected")

    conf = "high" if candidates["has_cut_word"] else "medium"
    return new_cut, best_str, True, reasons, conf


def generate_digit_confusions(n: int) -> set[int]:
    s = str(n)
    out = set()
    for i, ch in enumerate(s):
        if ch not in DIGIT_CONFUSION:
            continue
        for rep in DIGIT_CONFUSION[ch]:
            t = s[:i] + rep + s[i+1:]
            if t.isdigit():
                out.add(int(t))
    return out


def repair_new_cut_with_sequence(
    *,
    previous_cut_num: Optional[int],
    page: int,
    row: int,
    cut_num: Optional[int],
    cut_num_str: Optional[str],
    is_new_cut: bool,
    conf: str,
    reasons: List[str],
) -> Tuple[Optional[int], Optional[str], bool, List[str], str]:
    if previous_cut_num is None or cut_num is None or not is_new_cut:
        return cut_num, cut_num_str, is_new_cut, reasons, conf

    expected = previous_cut_num + 1
    if cut_num == expected:
        return cut_num, cut_num_str, is_new_cut, reasons, conf

    confusions = generate_digit_confusions(cut_num)
    if expected in confusions:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (digit_confusion)"]
        return expected, None, True, reasons, conf

    if conf != "high" and abs(cut_num - expected) >= 7:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (jump_too_large)"]
        return expected, None, True, reasons, conf

    if row == 1 and conf != "high" and abs(cut_num - expected) >= 5:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (page_row1_bias)"]
        return expected, None, True, reasons, conf

    return cut_num, cut_num_str, is_new_cut, reasons, conf


def main():
    episode_dir = CFG["episode_dir"]
    episode_id = CFG["episode_id"]

    out_root = os.path.join(CFG["out_dir"], episode_id)
    cuts_out_dir = os.path.join(out_root, "cuts")
    ensure_dir(out_root)
    ensure_dir(cuts_out_dir)

    # Load meta info for prompts
    symbol_terms_by_cat = load_symbol_terms_by_category(CFG["symbol_lexicon_path"])

    lexicon_path = os.path.join("outputs", CFG["script_phase"], CFG["episode_id"], CFG["lexicon_path"])
    lexicon_characters = load_lexicon_characters(lexicon_path)

    # Build per-column prompts (ready for later tuning)
    column_prompts: Dict[str, str] = {}
    for col in CFG["ocr_columns"]:
        column_prompts[col] = build_column_prompt(
            col=col,
            lexicon_characters=lexicon_characters,
            symbol_terms_by_cat=symbol_terms_by_cat,
        )

    # Load model once (base or LoRA adapter)
    model, processor = load_model_and_processor()

    # Select pages
    all_pages = list_pages(episode_dir)
    if CFG["page_select"] == "all":
        pages = all_pages
    else:
        pages = [p for p in all_pages if CFG["start_page"] <= p <= CFG["end_page"]]
    if not pages:
        raise RuntimeError("No pages selected/found.")
    
    first_page = pages[0]

    rows_per_page = CFG["rows_per_page"]
    cols_to_ocr = CFG["ocr_columns"]
    if "cut" not in cols_to_ocr:
        raise RuntimeError("CFG['ocr_columns'] must include 'cut'.")

    # OCR cache keyed by (path, col)
    ocr_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def ocr_path(path: str, col: str) -> Dict[str, Any]:
        key = (path, col)
        if key in ocr_cache:
            return ocr_cache[key]
        img = load_image(path)
        prompt = column_prompts.get(col, COLUMN_BASE_PROMPTS.get(col, "You are an OCR engine.\n" + BASE_RULES))

        # print(col, " prompt = \n", prompt, "\n")

        res = ocr_one_image(model, processor, img, prompt)
        ocr_cache[key] = res
        return res

    row_store: Dict[Tuple[int, int], Dict[str, Any]] = {}

    cut_spans: List[CutSpan] = []
    current: Optional[CutSpan] = None
    previous_cut_num: Optional[int] = None
    previous_cut_num_str: Optional[str] = None

    for page in tqdm(pages, desc="Pages"):
        full_page_img = page_image_path(episode_dir, page)

        for row in tqdm(range(1, rows_per_page + 1), desc=f"Page {page} rows", leave=False):
            key = (page, row)
            row_store[key] = {"page_image": full_page_img, "cols": {}}

            # CUT OCR + boundary decision
            cut_img_path = crop_path(episode_dir, "cut", page, row)
            if os.path.exists(cut_img_path):
                cut_ocr = ocr_path(cut_img_path, "cut")
                row_store[key]["cols"]["cut"] = {"image": cut_img_path, **cut_ocr}
                cand = extract_cut_candidates(cut_ocr["raw_text"], cut_ocr["lines"])
            else:
                cut_ocr = {"raw_text": "", "lines": [], "script": "unknown", "notes": [f"missing_image: cut/page{page}_row{row}"]}
                row_store[key]["cols"]["cut"] = {"image": None, **cut_ocr}
                cand = extract_cut_candidates("", [])

            cut_num, cut_num_str, is_new_cut, reasons, conf = decide_cut_transition(
                previous_cut_num=previous_cut_num,
                candidates=cand,
                page=page,
                row=row,
            )

            if page == first_page and row == 1:
                # OCRが 1 を含んでいない、または 0/0J のように怪しい場合だけ強制
                bad_first = (cut_num is None) or (cut_num != 1)
                if bad_first:
                    cut_num = 1
                    cut_num_str = "1"
                    is_new_cut = True
                    reasons = (reasons or []) + ["force_first_cut_to_1"]
                    conf = "forced"

            cut_num, cut_num_str, is_new_cut, reasons, conf = repair_new_cut_with_sequence(
                previous_cut_num=previous_cut_num,
                page=page,
                row=row,
                cut_num=cut_num,
                cut_num_str=cut_num_str,
                is_new_cut=is_new_cut,
                conf=conf,
                reasons=reasons,
            )

            if cut_num is not None:
                if cut_num == previous_cut_num and cut_num_str:
                    if previous_cut_num_str is None or len(cut_num_str) > len(previous_cut_num_str):
                        previous_cut_num_str = cut_num_str
                elif is_new_cut:
                    previous_cut_num_str = cut_num_str

            if is_new_cut and cut_num is not None:
                if current is not None:
                    prev_page, prev_row = page, row - 1
                    if prev_row < 1:
                        prev_page = page - 1
                        prev_row = rows_per_page
                    current.end = (prev_page, prev_row)
                    cut_spans.append(current)

                anchor = {
                    "page": page,
                    "row": row,
                    "image": cut_img_path if os.path.exists(cut_img_path) else None,
                    "ocr_raw_text": row_store[key]["cols"]["cut"]["raw_text"],
                    "ocr_lines": row_store[key]["cols"]["cut"]["lines"],
                    "candidates": {
                        "numbers": cand["numbers"],
                        "num_strs": cand["num_strs"],
                        "explicit_cut_str": cand["explicit_cut_str"],
                        "has_time": cand["has_time"],
                        "has_continue_word": cand["has_continue_word"],
                        "has_cut_word": cand["has_cut_word"],
                        "normalized_text": cand["normalized_text"],
                    },
                    "decision": "new_cut",
                    "decision_reason": reasons,
                    "confidence": conf,
                    "cut_number": cut_num,
                    "cut_number_str": cut_num_str,
                }
                current = CutSpan(
                    cut_num=cut_num,
                    cut_num_str=cut_num_str,
                    start=(page, row),
                    end=(page, row),
                    anchor=anchor,
                )
                previous_cut_num = cut_num
            else:
                if current is None:
                    anchor = {
                        "page": page,
                        "row": row,
                        "image": cut_img_path if os.path.exists(cut_img_path) else None,
                        "ocr_raw_text": row_store[key]["cols"]["cut"]["raw_text"],
                        "ocr_lines": row_store[key]["cols"]["cut"]["lines"],
                        "candidates": {
                            "numbers": cand["numbers"],
                            "num_strs": cand["num_strs"],
                            "explicit_cut_str": cand["explicit_cut_str"],
                            "has_time": cand["has_time"],
                            "has_continue_word": cand["has_continue_word"],
                            "has_cut_word": cand["has_cut_word"],
                            "normalized_text": cand["normalized_text"],
                        },
                        "decision": "start_unknown_cut",
                        "decision_reason": reasons,
                        "confidence": conf,
                        "cut_number": None,
                        "cut_number_str": None,
                    }
                    current = CutSpan(
                        cut_num=None,
                        cut_num_str=None,
                        start=(page, row),
                        end=(page, row),
                        anchor=anchor,
                    )

            # OCR other columns (column-specific prompts)
            for col in cols_to_ocr:
                if col == "cut":
                    continue
                img_path = crop_path(episode_dir, col, page, row)
                if not os.path.exists(img_path):
                    row_store[key]["cols"][col] = {
                        "image": None, "raw_text": "", "lines": [], "script": "unknown",
                        "notes": [f"missing_image: {col}/page{page}_row{row}"]
                    }
                    continue
                o = ocr_path(img_path, col)
                row_store[key]["cols"][col] = {"image": img_path, **o}

            if CFG["always_store_picture_path"] and "picture" not in row_store[key]["cols"]:
                pic_path = crop_path(episode_dir, "picture", page, row)
                row_store[key]["cols"]["picture"] = {"image": pic_path if os.path.exists(pic_path) else None}

            if current is not None:
                current.end = (page, row)

    if current is not None:
        cut_spans.append(current)

    if not cut_spans:
        raise RuntimeError("No cut spans created. Check inputs and cut/ crops.")

    index: Dict[str, Any] = {
        "episode_id": episode_id,
        "episode_dir": episode_dir,
        "page_select": {k: CFG[k] for k in ["page_select", "start_page", "end_page"] if k in CFG},
        "rows_per_page": CFG["rows_per_page"],
        "ocr_columns": CFG["ocr_columns"],
        "cuts": [],
        "meta": {
            "model_id": CFG["model_id"],
            "created_at": now_iso_utc(),
            "gen": CFG["gen"],
            "use_lora_adapter": CFG["use_lora_adapter"],
            "lora_adapter_path": CFG["lora_adapter_path"] if CFG["use_lora_adapter"] else None,
            "symbol_lexicon_path": CFG["symbol_lexicon_path"],
            "lexicon_path": lexicon_path,
            "lexicon_character_count": len(lexicon_characters),
            "term_categories_used_by_column": COLUMN_TERM_CATEGORIES,
        },
    }

    for span in cut_spans:
        start_page, start_row = span.start
        end_page, end_row = span.end

        keys: List[Tuple[int, int]] = []
        for p in range(start_page, end_page + 1):
            r0 = start_row if p == start_page else 1
            r1 = end_row if p == end_page else CFG["rows_per_page"]
            for r in range(r0, r1 + 1):
                keys.append((p, r))

        if span.cut_num is not None:
            out_name = f"cut{span.cut_num:04d}.stage1.json"
        else:
            out_name = f"cut_unknown_p{start_page}_r{start_row}.stage1.json"
        out_path = os.path.join(cuts_out_dir, out_name)

        if CFG["skip_if_exists"] and os.path.exists(out_path):
            index["cuts"].append({
                "cut": span.cut_num,
                "cut_str": span.cut_num_str,
                "span": {"start": {"page": start_page, "row": start_row}, "end": {"page": end_page, "row": end_row}},
                "out_json": out_path,
                "skipped": True,
            })
            continue

        rows_payload: List[Dict[str, Any]] = []
        for (p, r) in keys:
            rs = row_store.get((p, r))
            if rs is None:
                continue
            rows_payload.append({
                "page": p,
                "row": r,
                "page_image": rs["page_image"],
                "cols": rs["cols"],
            })

        cut_obj: Dict[str, Any] = {
            "episode_id": episode_id,
            "cut": span.cut_num,
            "cut_str": span.cut_num_str,
            "span": {
                "start": {"page": start_page, "row": start_row},
                "end": {"page": end_page, "row": end_row},
            },
            "anchor": span.anchor,
            "rows": rows_payload,
            "meta": {
                "created_at": now_iso_utc(),
                "model_id": CFG["model_id"],
                "gen": CFG["gen"],
                "use_lora_adapter": CFG["use_lora_adapter"],
                "lora_adapter_path": CFG["lora_adapter_path"] if CFG["use_lora_adapter"] else None,
            },
            "notes": [
                "stage1_ocr_only",
                "cross_pages" if start_page != end_page else "within_page",
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cut_obj, f, ensure_ascii=False, indent=2)

        index["cuts"].append({
            "cut": span.cut_num,
            "cut_str": span.cut_num_str,
            "span": {"start": {"page": start_page, "row": start_row}, "end": {"page": end_page, "row": end_row}},
            "out_json": out_path,
            "skipped": False,
        })

        print(f"[OK] wrote {out_path}  span=({start_page},{start_row})..({end_page},{end_row})")

    index_path = os.path.join(out_root, "index.stage1.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {index_path}")


if __name__ == "__main__":
    main()
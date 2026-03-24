# stage1_ocr_extract_episode01_hybrid_seqfix.py
#
# Stage1 (VLM OCR):
# - Extract cuts + OCR per column (cut/picture/action_memo/dialogue/time)
# - Column-wise BASE / LoRA switching (disable_adapter for base)
# - Cut number sequence correction (online + postprocess)
# - Force first cut = 1 (configurable)
# - Episode fixed to episode01 (data/episode01)
#
# Output:
# - outputs/episode01/cuts/cutXXXX.stage1.json
# - outputs/episode01/index.stage1.json
#
# Notes:
# - This keeps the original stage1 JSON schema and only adds extra keys (model_variant etc).
# - Postprocess sequence correction aims to prevent:
#     * misread like cut60 -> cut69
#     * non-monotonic / large jumps
#   It DOES NOT magically split a cut span if the boundary itself was not detected.
#   (If you want boundary recovery, we can add a more aggressive heuristic separately.)

import os
import re
import json
import hashlib
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
    # -------------------------
    # Input (FIXED: episode01)
    # -------------------------
    "episode_id": "free2",
    "episode_dir": "data/free/free2",
    "script_phase": "script_phase1",

    # Output base dir (stage1 outputs)
    "out_dir": "outputs_free2",

    # Page selection
    "page_select": "all",  # "all" or "range"
    "start_page": 2,
    "end_page": 109,

    # Usually 5 rows per storyboard page in your preprocessing
    "rows_per_page": 5,

    # OCR columns
    "ocr_columns": ["cut", "picture", "action_memo", "dialogue", "time"],

    # -------------------------
    # Model
    # -------------------------
    "base_model_id": "Qwen/Qwen3-VL-4B-Instruct",

    # LoRA adapter directory (set to None or "" to disable)
    # e.g. "outputs_ft/stage1_ocr_lora_.../adapter"
    "lora_adapter_dir": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train50per/adapter",

    # Column-wise policy:
    # - use LoRA for handwriting-heavy columns
    # - keep cut/time as base if LoRA harms them
    "use_lora_for_columns": {"picture", "action_memo", "dialogue"},
    "force_base_for_columns": {"cut", "time"},

    # Torch
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,

    # Generation defaults (deterministic)
    "gen_default": {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    },

    # IMPORTANT: per-column max tokens (reduce over-generation)
    "max_new_tokens_by_col": {
        "cut": 16,
        "time": 16,
        "picture": 96,
        "action_memo": 256,
        "dialogue": 256,
    },

    # If True, skip writing when a cut file already exists
    "skip_if_exists": False,

    # If picture is not OCRed, still store picture image path
    "always_store_picture_path": True,

    # Lexicons
    "symbol_lexicon_path": "data/storyboard_symbol_lexicon.json",
    # stage0 output: outputs/script_phase1/episode01/lexicon_entities.json
    "lexicon_path": "lexicon_entities.json",

    # Prompt injection limits (avoid too long prompts)
    "prompt_max_char_names": 200,
    "prompt_max_terms_per_category": 60,
    "prompt_max_categories": 12,

    # -------------------------
    # Cut sequence correction
    # -------------------------
    # Force first cut number
    "force_first_cut": True,
    "start_cut_number": 1,

    # Online correction thresholds (inside loop)
    "online_jump_repair_threshold": 3,   # if abs(pred-expected) >= 7 and not high confidence, repair
    "online_row1_jump_repair_threshold": 5,

    # Postprocess (2-pass) correction (after building spans)
    "enable_post_sequence_fix": True,
    "post_max_jump_allowed": 3,          # if pred > expected + this, override to expected
    "post_allow_small_offset": 0,        # allow pred==expected+offset (0 means strict)
}
# =========================


# -------------------------
# Column-specific term-category selection
# -------------------------
COLUMN_TERM_CATEGORIES = {
    "cut": ["structure", "editing", "timing"],
    "picture": ["camera_movement", "camera_angle", "framing", "focus", "timing", "editing"],
    "action_memo": [
        "camera_movement", "camera_angle", "framing", "focus",
        "timing", "effect", "animation", "editing", "structure", "production", "expression", "sound", "dialogue"
    ],
    "dialogue": ["dialogue", "sound", "editing", "structure"],
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
    "- Return ONLY the transcribed text.\n"
    "- Do NOT add any labels or explanations.\n"
)

HANDWRITING_NOTE = (
    "Notes:\n"
    "- This storyboard contains BOTH handwritten text and computer-typed text.\n"
    "- Handwritten text may include arrows and symbols.\n"
    "- Arrows are important. Always transcribe arrows if present.\n\n"
)

STRICT_FOOTER = (
    "Output format:\n"
    "- Return ONLY the transcribed text.\n"
    "- Do NOT add labels like 'Transcription:' or '補正後テキスト:'.\n"
    "- Do NOT add explanations.\n"
)

COLUMN_BASE_PROMPTS = {
    "cut": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: CUT.\n"
        "Task: Read the cut number if present. It may also include a time like mm:ss.\n"
        "Handwritten digits may appear.\n"
        "If nothing readable, output '□'.\n\n"
        + BASE_RULES + "\n"
        + HANDWRITING_NOTE +
        "Extra constraints for CUT column:\n"
        "- Prefer digits and very short markers.\n"
        "- Do NOT guess missing digits.\n"
        "- If output includes non-digit junk, output '□' instead.\n\n"
        + STRICT_FOOTER + "\n"
    ),
    "picture": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: PICTURE (mostly drawings).\n"
        "Task: Transcribe ANY text that appears inside the picture area.\n"
        "Most of the region is drawings, but sometimes handwritten terms exist.\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, punctuation).\n"
        "If there is no text, output '□'.\n\n"
        + BASE_RULES + "\n"
        + HANDWRITING_NOTE +
        "Extra constraints for PICTURE column:\n"
        "- Only transcribe visible text/symbols; ignore the drawings.\n"
        "- If arrows are attached to a term (e.g., →PAN), keep them together.\n\n"
        + STRICT_FOOTER + "\n"
    ),
    "action_memo": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: ACTION MEMO.\n"
        "Task: Transcribe all handwritten/typed notes about action, camera, timing, and effects.\n"
        "This column often contains many production terms.\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, punctuation).\n\n"
        + BASE_RULES + "\n"
        + HANDWRITING_NOTE +
        "Extra constraints for ACTION MEMO column:\n"
        "- Preserve production terms exactly.\n"
        "- Keep parentheses and arrows exactly.\n"
        "- Do NOT normalize or rewrite.\n\n"
        + STRICT_FOOTER + "\n"
    ),
    "dialogue": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: DIALOGUE.\n"
        "Task: Transcribe dialogue lines and related audio notes.\n"
        "This column may include speaker names, quoted lines, and sound notes.\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, punctuation).\n\n"
        + BASE_RULES + "\n"
        + HANDWRITING_NOTE +
        "Extra constraints for DIALOGUE column:\n"
        "- Preserve speaker names and symbols.\n"
        "- Preserve audio tags like SE, MA, BGM, ガヤ, off/on.\n\n"
        + STRICT_FOOTER + "\n"
    ),
    "time": (
        "You are an OCR engine for anime storyboards.\n"
        "Target column: TIME.\n"
        "Task: Transcribe time information if present.\n"
        "Common formats: mm:ss, frame counts, short numeric timing notes.\n"
        "If there is no readable timing, output '□'.\n\n"
        + BASE_RULES + "\n"
        + HANDWRITING_NOTE +
        "Extra constraints for TIME column:\n"
        "- Prefer numeric/time expressions.\n"
        "- Keep exactly as written.\n\n"
        + STRICT_FOOTER + "\n"
    ),
}


# -------------------------
# Lexicon loaders
# -------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    chars = sorted(set(chars), key=lambda x: (-len(x), x))
    return chars


def load_symbol_terms_by_category(path_json: str) -> Dict[str, List[str]]:
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
    for cat in cats[: int(CFG["prompt_max_categories"])]:
        terms = symbol_terms_by_cat.get(cat, [])
        if not terms:
            continue
        lines.append(f"[{cat}]")
        lines.extend(terms[: int(CFG["prompt_max_terms_per_category"])])
    return "\n".join(lines) + "\n\n"


def make_character_block_for_column(lexicon_characters: List[str], col: str) -> str:
    if col not in {"dialogue", "action_memo"}:
        return ""
    if not lexicon_characters:
        return ""
    return (
        "=== CHARACTER NAMES (terms only) ===\n"
        + "\n".join(lexicon_characters[: int(CFG["prompt_max_char_names"])])
        + "\n\n"
    )


def build_column_prompt(col: str, lexicon_characters: List[str], symbol_terms_by_cat: Dict[str, List[str]]) -> str:
    base = COLUMN_BASE_PROMPTS.get(col, "You are an OCR engine.\n" + BASE_RULES + "\n")
    char_block = make_character_block_for_column(lexicon_characters, col)
    terms_block = make_terms_block_for_column(symbol_terms_by_cat, col)
    return base + char_block + terms_block + STRICT_FOOTER + "\n"
    # return base + STRICT_FOOTER + "\n"


# -------------------------
# File patterns / helpers
# -------------------------
PAGE_RE = re.compile(r"^page(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
ROW_IMG_NAME = "page{page}_row{row}.png"

TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
CONTINUE_WORD_RE = re.compile(r"(続|つづく|継続|続き|つづき|CONT|CONTINUED)", re.IGNORECASE)
CUT_WORD_RE = re.compile(r"(CUT|ＣＵＴ|C\s*U\s*T)", re.IGNORECASE)

# digit confusions (for cut numbers)
DIGIT_CONFUSION = {
    "0": ["9", "8", "6"],
    "9": ["0"],
    "8": ["0"],
    "6": ["0", "5"],
    "1": ["7", "J"],
    "7": ["1"],
    "2": ["3"],
    "3": ["2"],
    "5": ["6"],
    "J": ["1"],
}


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


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


def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]


# -------------------------
# Model loading + column-wise base/LoRA switch
# -------------------------
def load_model_and_processor():
    kwargs = {
        "device_map": CFG["device_map"],
        "torch_dtype": CFG["torch_dtype"],
    }
    if CFG["flash_attn2"]:
        kwargs["attn_implementation"] = "flash_attention_2"

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(CFG["base_model_id"], **kwargs)
    processor = AutoProcessor.from_pretrained(CFG["base_model_id"])
    base_model.eval()

    # Ensure pad_token exists
    tok = processor.tokenizer
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    lora_dir = CFG.get("lora_adapter_dir")
    if lora_dir and os.path.isdir(lora_dir):
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapter, but not installed.")
        model = PeftModel.from_pretrained(base_model, lora_dir)
        model.eval()
        return model, processor, True
    return base_model, processor, False


def column_uses_lora(col: str, has_lora: bool) -> bool:
    if not has_lora:
        return False
    if col in (CFG.get("force_base_for_columns") or set()):
        return False
    return col in (CFG.get("use_lora_for_columns") or set())


def max_new_tokens_for_col(col: str) -> int:
    return int((CFG.get("max_new_tokens_by_col") or {}).get(col, 256))


@torch.no_grad()
def ocr_one_image(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    *,
    col: str,
    use_lora: bool,
) -> Dict[str, Any]:
    """
    OCR as plain text (no JSON).
    If model is PeftModel and use_lora=False => generate under disable_adapter() = base behavior.
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

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    gen = dict(CFG["gen_default"])
    gen["max_new_tokens"] = max_new_tokens_for_col(col)
    gen["pad_token_id"] = pad_id
    if eos_id is not None:
        gen["eos_token_id"] = eos_id

    if hasattr(model, "disable_adapter") and (not use_lora):
        with model.disable_adapter():
            out_ids = model.generate(**inputs, **gen)
        variant = "base"
    else:
        out_ids = model.generate(**inputs, **gen)
        variant = "lora" if use_lora else "base"

    trimmed = out_ids[0][len(inputs["input_ids"][0]):]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()

    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    return {
        "raw_text": text,
        "lines": lines,
        "model_variant": variant,
        "prompt_hash": _prompt_hash(prompt_text),
        "notes": [],
    }


# -------------------------
# Cut boundary heuristics
# -------------------------
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
        try:
            n = int(s)
        except Exception:
            continue
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
    """
    Return:
      (cut_num_pred, cut_num_str, is_new_cut, reasons, confidence)
    """
    reasons: List[str] = []
    nums: List[int] = candidates["numbers"]
    num_strs: List[str] = candidates["num_strs"]
    explicit_cut_str: Optional[str] = candidates["explicit_cut_str"]

    # If no numbers at all, treat as continuation (same behavior as original),
    # because many rows inside the same cut can have empty cut-number cells.
    if not nums:
        return previous_cut_num, None, False, ["no_numeric_token"], "none"

    # If previous cut number appears among extracted numbers -> likely continuation
    if previous_cut_num is not None and previous_cut_num in nums:
        reasons.append("same_as_previous_cut")
        if candidates["has_continue_word"]:
            reasons.append("explicit_continue_word")
        if row == 1:
            reasons.append("page_start_row")

        best_str = None
        for s in num_strs:
            try:
                if int(s) == previous_cut_num:
                    if best_str is None or len(s) > len(best_str):
                        best_str = s
            except Exception:
                pass
        if best_str:
            reasons.append("preserve_zero_padded_str")
        return previous_cut_num, best_str, False, reasons, "high" if candidates["has_continue_word"] else "medium"

    # Otherwise new cut (pick first number)
    new_cut = nums[0]
    reasons.append("new_cut_number")

    best_str = None
    if explicit_cut_str:
        try:
            if int(explicit_cut_str) == new_cut:
                best_str = explicit_cut_str
                reasons.append("explicit_cut_form")
        except Exception:
            pass

    if best_str is None:
        for s in num_strs:
            try:
                if int(s) == new_cut:
                    if best_str is None or len(s) > len(best_str):
                        best_str = s
            except Exception:
                pass

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
            t = s[:i] + rep + s[i + 1:]
            # allow strings like "0J"? -> ignore non-digit
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
    """
    Online repair to reduce catastrophic jumps.
    """
    if previous_cut_num is None or cut_num is None or not is_new_cut:
        return cut_num, cut_num_str, is_new_cut, reasons, conf

    # sanitize invalid
    if cut_num <= 0:
        expected = previous_cut_num + 1
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (non_positive)"]
        return expected, None, True, reasons, conf

    expected = previous_cut_num + 1
    if cut_num == expected:
        return cut_num, cut_num_str, is_new_cut, reasons, conf

    # digit confusion repair (works even if conf is high)
    confusions = generate_digit_confusions(cut_num)
    if expected in confusions:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (digit_confusion)"]
        return expected, None, True, reasons, conf

    # big jump repair if not high confidence
    if conf != "high" and abs(cut_num - expected) >= int(CFG["online_jump_repair_threshold"]):
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (jump_too_large)"]
        return expected, None, True, reasons, conf

    # page-row1 bias
    if row == 1 and conf != "high" and abs(cut_num - expected) >= int(CFG["online_row1_jump_repair_threshold"]):
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (page_row1_bias)"]
        return expected, None, True, reasons, conf

    return cut_num, cut_num_str, is_new_cut, reasons, conf


def postprocess_cut_spans_sequence(
    cut_spans: List[CutSpan],
    *,
    start_cut: int,
    max_jump_allowed: int,
    allow_small_offset: int,
) -> List[CutSpan]:
    """
    2-pass correction:
    - enforce first cut = start_cut
    - enforce monotonic increasing
    - clamp suspicious large jumps to expected (=prev+1)
    """
    if not cut_spans:
        return cut_spans

    # already in order, but keep safe
    spans = sorted(cut_spans, key=lambda s: (s.start[0], s.start[1]))

    prev_final = start_cut - 1
    for i, s in enumerate(spans):
        pred = s.cut_num
        expected = start_cut if i == 0 else (prev_final + 1)

        # store prediction
        s.anchor["cut_number_pred_post"] = pred
        s.anchor["cut_number_expected_post"] = expected

        # force first
        if i == 0 and CFG.get("force_first_cut", True):
            final = start_cut
            reason = "post_force_first_cut"
        else:
            if pred is None:
                final = expected
                reason = "post_fill_missing_pred"
            else:
                # invalid / non-positive
                if pred <= 0:
                    final = expected
                    reason = "post_non_positive_to_expected"
                # going backwards or duplicate while being a new span
                elif pred <= prev_final:
                    final = expected
                    reason = "post_non_monotonic_to_expected"
                # allow small offset
                elif pred <= expected + allow_small_offset:
                    final = pred
                    reason = "post_accept_pred_close_to_expected"
                # huge jump
                elif pred > expected + max_jump_allowed:
                    final = expected
                    reason = "post_clamp_huge_jump_to_expected"
                else:
                    final = pred
                    reason = "post_accept_pred"

        # apply
        if final != pred:
            s.anchor["cut_number_post_fix"] = {"from": pred, "to": final, "reason": reason}
        else:
            s.anchor["cut_number_post_fix"] = {"from": pred, "to": final, "reason": reason}

        s.cut_num = final
        s.cut_num_str = str(final)
        prev_final = final

    return spans


# -------------------------
# Main
# -------------------------
def main():
    episode_dir = CFG["episode_dir"]
    episode_id = CFG["episode_id"]

    out_root = os.path.join(CFG["out_dir"], episode_id)
    cuts_out_dir = os.path.join(out_root, "cuts")
    ensure_dir(out_root)
    ensure_dir(cuts_out_dir)

    # Load meta info for prompts
    symbol_terms_by_cat = load_symbol_terms_by_category(CFG["symbol_lexicon_path"])

    # stage0 lexicon path
    lexicon_path = os.path.join(CFG["out_dir"], CFG["script_phase"], CFG["episode_id"], CFG["lexicon_path"])
    lexicon_characters = load_lexicon_characters(lexicon_path)

    # Build per-column prompts
    column_prompts: Dict[str, str] = {}
    for col in CFG["ocr_columns"]:
        column_prompts[col] = build_column_prompt(
            col=col,
            lexicon_characters=lexicon_characters,
            symbol_terms_by_cat=symbol_terms_by_cat,
        )

    # Load model once (base + optional LoRA adapter)
    model, processor, has_lora = load_model_and_processor()

    # Select pages
    all_pages = list_pages(episode_dir)
    if CFG["page_select"] == "all":
        pages = all_pages
    else:
        pages = [p for p in all_pages if int(CFG["start_page"]) <= p <= int(CFG["end_page"])]
    if not pages:
        raise RuntimeError("No pages selected/found.")

    first_page = pages[0]

    rows_per_page = int(CFG["rows_per_page"])
    cols_to_ocr = list(CFG["ocr_columns"])
    if "cut" not in cols_to_ocr:
        raise RuntimeError("CFG['ocr_columns'] must include 'cut'.")

    # OCR cache keyed by (path, col, variant, prompt_hash)
    ocr_cache: Dict[str, Dict[str, Any]] = {}

    def ocr_path(path: str, col: str) -> Dict[str, Any]:
        use_lora = column_uses_lora(col, has_lora)
        prompt = column_prompts.get(col, COLUMN_BASE_PROMPTS.get(col, "You are an OCR engine.\n" + BASE_RULES))
        key = f"{'lora' if use_lora else 'base'}:{col}:{_prompt_hash(prompt)}:{path}"
        if key in ocr_cache:
            return ocr_cache[key]
        img = load_image(path)
        # print(prompt)
        res = ocr_one_image(model, processor, img, prompt, col=col, use_lora=use_lora)
        ocr_cache[key] = res
        return res

    row_store: Dict[Tuple[int, int], Dict[str, Any]] = {}

    cut_spans: List[CutSpan] = []
    current: Optional[CutSpan] = None
    previous_cut_num: Optional[int] = None
    previous_cut_num_str: Optional[str] = None

    # Iterate pages/rows
    for page in tqdm(pages, desc="Pages"):
        full_page_img = page_image_path(episode_dir, page)

        for row in tqdm(range(1, rows_per_page + 1), desc=f"Page {page} rows", leave=False):
            key = (page, row)
            row_store[key] = {"page_image": full_page_img, "cols": {}}

            # --- CUT OCR + boundary decision ---
            cut_img_path = crop_path(episode_dir, "cut", page, row)
            if os.path.exists(cut_img_path):
                cut_ocr = ocr_path(cut_img_path, "cut")
                row_store[key]["cols"]["cut"] = {"image": cut_img_path, **cut_ocr}
                cand = extract_cut_candidates(cut_ocr["raw_text"], cut_ocr["lines"])
            else:
                cut_ocr = {"raw_text": "", "lines": [], "model_variant": "none", "prompt_hash": "", "notes": [f"missing_image: cut/page{page}_row{row}"]}
                row_store[key]["cols"]["cut"] = {"image": None, **cut_ocr}
                cand = extract_cut_candidates("", [])

            cut_num, cut_num_str, is_new_cut, reasons, conf = decide_cut_transition(
                previous_cut_num=previous_cut_num,
                candidates=cand,
                page=page,
                row=row,
            )

            # Force first cut
            if CFG.get("force_first_cut", True) and page == first_page and row == 1:
                start_cut = int(CFG.get("start_cut_number", 1))
                if cut_num != start_cut:
                    cut_num = start_cut
                    cut_num_str = str(start_cut)
                    is_new_cut = True
                    reasons = (reasons or []) + [f"force_first_cut_to_{start_cut}"]
                    conf = "forced"

            # Online sequence repair (reduce huge errors)
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

            # Track best string for same cut
            if cut_num is not None:
                if cut_num == previous_cut_num and cut_num_str:
                    if previous_cut_num_str is None or len(cut_num_str) > len(previous_cut_num_str):
                        previous_cut_num_str = cut_num_str
                elif is_new_cut:
                    previous_cut_num_str = cut_num_str

            # Create / close cut spans
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
                    "ocr_raw_text": row_store[key]["cols"]["cut"].get("raw_text", ""),
                    "ocr_lines": row_store[key]["cols"]["cut"].get("lines", []),
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
                    "cut_number_pred": cut_num,
                    "cut_number_str_pred": cut_num_str,
                    "cut_ocr_model_variant": row_store[key]["cols"]["cut"].get("model_variant"),
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
                    # start unknown span
                    anchor = {
                        "page": page,
                        "row": row,
                        "image": cut_img_path if os.path.exists(cut_img_path) else None,
                        "ocr_raw_text": row_store[key]["cols"]["cut"].get("raw_text", ""),
                        "ocr_lines": row_store[key]["cols"]["cut"].get("lines", []),
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
                        "cut_number_pred": None,
                        "cut_number_str_pred": None,
                        "cut_ocr_model_variant": row_store[key]["cols"]["cut"].get("model_variant"),
                    }
                    current = CutSpan(
                        cut_num=None,
                        cut_num_str=None,
                        start=(page, row),
                        end=(page, row),
                        anchor=anchor,
                    )

            # --- OCR other columns (column-wise base/LoRA) ---
            for col in cols_to_ocr:
                if col == "cut":
                    continue
                img_path = crop_path(episode_dir, col, page, row)
                if not os.path.exists(img_path):
                    row_store[key]["cols"][col] = {
                        "image": None,
                        "raw_text": "",
                        "lines": [],
                        "model_variant": "none",
                        "prompt_hash": "",
                        "notes": [f"missing_image: {col}/page{page}_row{row}"],
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
        raise RuntimeError("No cut spans created. Check inputs and cut crops.")

    # -------------------------
    # Post sequence fix (2-pass)
    # -------------------------
    if CFG.get("enable_post_sequence_fix", True):
        cut_spans = postprocess_cut_spans_sequence(
            cut_spans,
            start_cut=int(CFG.get("start_cut_number", 1)),
            max_jump_allowed=int(CFG.get("post_max_jump_allowed", 3)),
            allow_small_offset=int(CFG.get("post_allow_small_offset", 0)),
        )

    # -------------------------
    # Write outputs
    # -------------------------
    index: Dict[str, Any] = {
        "episode_id": episode_id,
        "episode_dir": episode_dir,
        "page_select": {k: CFG[k] for k in ["page_select", "start_page", "end_page"] if k in CFG},
        "rows_per_page": rows_per_page,
        "ocr_columns": cols_to_ocr,
        "cuts": [],
        "meta": {
            "created_at": now_iso_utc(),
            "base_model_id": CFG["base_model_id"],
            "lora_adapter_dir": CFG.get("lora_adapter_dir"),
            "has_lora": has_lora,
            "use_lora_for_columns": sorted(list(CFG.get("use_lora_for_columns") or [])),
            "force_base_for_columns": sorted(list(CFG.get("force_base_for_columns") or [])),
            "gen_default": CFG["gen_default"],
            "max_new_tokens_by_col": CFG["max_new_tokens_by_col"],
            "symbol_lexicon_path": CFG["symbol_lexicon_path"],
            "lexicon_path": lexicon_path,
            "lexicon_character_count": len(lexicon_characters),
            "term_categories_used_by_column": COLUMN_TERM_CATEGORIES,
            "sequence_fix": {
                "force_first_cut": CFG.get("force_first_cut", True),
                "start_cut_number": CFG.get("start_cut_number", 1),
                "enable_post_sequence_fix": CFG.get("enable_post_sequence_fix", True),
                "post_max_jump_allowed": CFG.get("post_max_jump_allowed", 3),
                "post_allow_small_offset": CFG.get("post_allow_small_offset", 0),
            }
        },
    }

    for span in cut_spans:
        start_page, start_row = span.start
        end_page, end_row = span.end

        keys: List[Tuple[int, int]] = []
        for p in range(start_page, end_page + 1):
            r0 = start_row if p == start_page else 1
            r1 = end_row if p == end_page else rows_per_page
            for r in range(r0, r1 + 1):
                keys.append((p, r))

        if span.cut_num is not None:
            out_name = f"cut{int(span.cut_num):04d}.stage1.json"
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
                "base_model_id": CFG["base_model_id"],
                "lora_adapter_dir": CFG.get("lora_adapter_dir"),
                "has_lora": has_lora,
                "use_lora_for_columns": sorted(list(CFG.get("use_lora_for_columns") or [])),
                "force_base_for_columns": sorted(list(CFG.get("force_base_for_columns") or [])),
                "gen_default": CFG["gen_default"],
                "max_new_tokens_by_col": CFG["max_new_tokens_by_col"],
            },
            "notes": [
                "stage1_ocr_only",
                "cross_pages" if start_page != end_page else "within_page",
                "columnwise_base_lora_switch",
                "sequence_fix_post" if CFG.get("enable_post_sequence_fix", True) else "sequence_fix_disabled",
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

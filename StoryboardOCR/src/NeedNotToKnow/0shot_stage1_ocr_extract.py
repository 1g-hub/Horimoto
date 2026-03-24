# stage1_extract_cuts.py

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

# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # Input
    "episode_dir": "data/episode01",  # "data/episode01"
    # Output base dir
    "out_dir": "outputs",

    # Page selection
    # - "all": use all pages found in pages/
    # - "range": use start_page..end_page (inclusive)
    "page_select": "all",  # "all" or "range"
    "start_page": 28,
    "end_page": 32,

    # Usually 5 rows per storyboard page in your preprocessing
    "rows_per_page": 5,

    # OCR columns (cut is mandatory for cut boundary decisions)
    # Add "picture" if you want OCR over picture crops too (slower; usually optional).
    "ocr_columns": ["cut", "picture", "action_memo", "dialogue", "time"],  # + "picture" if needed

    # Model
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "device_map": "auto",
    "torch_dtype": "auto",
    "flash_attn2": False,

    # Generation (keep minimal & deterministic for OCR)
    # NOTE: Some transformers versions warn that temperature/top_k might be ignored; avoid them.
    "gen": {
        "max_new_tokens": 512,
        "do_sample": False,
        # "temperature": 0.0,  # avoid to suppress "invalid flags" warnings across versions
        # "top_p": 1.0,
        # "top_k": 0,
        "num_beams": 1,
    },

    # If True, skip writing a cut file when it already exists
    "skip_if_exists": False,

    # If picture is not OCRed, we still store picture image path for Stage2
    "always_store_picture_path": True,
}
# =========================


# -------------------------
# Prompt (OCR only, text output)
# -------------------------
OCR_SYSTEM = (
    "You are an OCR engine for anime storyboards.\n"
    "Task: transcribe ALL visible text as faithfully as possible.\n"
    "Rules:\n"
    "- Keep original line breaks.\n"
    "- Do NOT correct typos.\n"
    "- Do NOT translate.\n"
    "- If unreadable, output '□'.\n"
    "Return ONLY the transcribed text. No explanations.\n"
)

# -------------------------
# File patterns / helpers
# -------------------------
PAGE_RE = re.compile(r"^page(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
ROW_IMG_NAME = "page{page}_row{row}.png"

# Time token (always NOT a cut number)
TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}\b")

# Continuation keywords (Japanese + English)
CONTINUE_WORD_RE = re.compile(r"(続|つづく|継続|続き|つづき|CONT|CONTINUED)", re.IGNORECASE)

# Optional "CUT" word (can boost confidence)
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


def load_model():
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
def ocr_one_image(model, processor, image: Image.Image) -> Dict[str, Any]:
    """
    OCR as plain text (no JSON) to avoid JSONDecode errors.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": OCR_SYSTEM},
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
    return {
        "raw_text": text,
        "lines": lines,
        "script": "unknown",
        "notes": [],
    }


def extract_cut_candidates(raw_text: str, lines: List[str]) -> Dict[str, Any]:
    """
    Extract candidate cut numbers from cut-column OCR.
    Supports:
      - "CUT 001"
      - "001"
      - "15\\n05:00"  (number + time in same crop)
      - "10 続"
      - "10 続\\n00:00"
    Returns both numeric and string forms (to preserve zero padding).
    """
    t = " ".join([raw_text] + (lines or []))
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    has_time = bool(TIME_TOKEN_RE.search(t))
    has_continue_word = bool(CONTINUE_WORD_RE.search(t))
    has_cut_word = bool(CUT_WORD_RE.search(t))

    # Remove time tokens for number extraction
    t_wo_time = TIME_TOKEN_RE.sub(" ", t)

    # Prefer explicit CUT patterns if present (keeps leading zeros)
    m = re.search(r"(CUT|ＣＵＴ|C\s*U\s*T)\s*([0-9]{1,4})", t_wo_time, flags=re.IGNORECASE)
    explicit_cut_str = m.group(2) if m else None

    # Extract number strings (keep zero padding)
    num_strs = re.findall(r"\b\d{1,4}\b", t_wo_time)

    # Numeric values (001 -> 1), keep order of appearance but de-dup
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
        "explicit_cut_str": explicit_cut_str,  # e.g., "001"
        "num_strs": num_strs,                  # e.g., ["001", "10"]
        "numbers": nums,                       # e.g., [1, 10]
    }


@dataclass
class CutSpan:
    cut_num: Optional[int]
    cut_num_str: Optional[str]          # preserves zero padding if observed (e.g., "001")
    start: Tuple[int, int]              # (page, row)
    end: Tuple[int, int]                # (page, row)
    anchor: Dict[str, Any]              # decision evidence


def decide_cut_transition(
    *,
    previous_cut_num: Optional[int],
    candidates: Dict[str, Any],
    page: int,
    row: int,
) -> Tuple[Optional[int], Optional[str], bool, List[str], str]:
    """
    Decide whether this row starts a new cut, or continues previous cut.
    Returns:
      (cut_num, cut_num_str, is_new_cut, reasons[], confidence)
    """
    reasons: List[str] = []
    nums: List[int] = candidates["numbers"]
    num_strs: List[str] = candidates["num_strs"]
    explicit_cut_str: Optional[str] = candidates["explicit_cut_str"]

    # No numeric token -> continue
    if not nums:
        return previous_cut_num, None, False, ["no_numeric_token"], "none"

    # If previous cut number appears in current text => continuation
    if previous_cut_num is not None and previous_cut_num in nums:
        reasons.append("same_as_previous_cut")
        if candidates["has_continue_word"]:
            reasons.append("explicit_continue_word")
        if row == 1:
            reasons.append("page_start_row")

        # Preserve best string representation for the previous cut number if present
        # Prefer exact match (with zeros) if any; else use explicit_cut_str if it normalizes to previous.
        best_str = None
        for s in num_strs:
            if int(s) == previous_cut_num:
                # choose the longest (most zero padded) representation
                if best_str is None or len(s) > len(best_str):
                    best_str = s
        if best_str:
            reasons.append("preserve_zero_padded_str")
        return previous_cut_num, best_str, False, reasons, "high" if candidates["has_continue_word"] else "medium"

    # Otherwise treat as new cut (first numeric token)
    new_cut = nums[0]
    reasons.append("new_cut_number")

    # Preserve string form for this new cut (prefer explicit CUT form; else longest matching token)
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

    # Confidence: higher if CUT word present, or continue word (rare for new) present
    conf = "high" if candidates["has_cut_word"] else "medium"
    return new_cut, best_str, True, reasons, conf


def generate_digit_confusions(n: int) -> set[int]:
    """1桁だけ置換した候補を生成（例: 69 -> 60 を作れる可能性）"""
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
    """
    new_cut と判定されたとき、prev+1 と整合するように補正する。
    返り値: (cut_num, cut_num_str, is_new_cut, reasons, conf)
    """
    if previous_cut_num is None or cut_num is None or not is_new_cut:
        return cut_num, cut_num_str, is_new_cut, reasons, conf

    expected = previous_cut_num + 1
    if cut_num == expected:
        return cut_num, cut_num_str, is_new_cut, reasons, conf

    # まず 1桁誤読で expected が生成できるならそれを採用（例: 69 -> 60）
    confusions = generate_digit_confusions(cut_num)
    if expected in confusions:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (digit_confusion)"]
        # strは正規化（ゼロ埋めは stage1では必須でないので None でもOK）
        return expected, None, True, reasons, conf

    # 「飛びすぎ」を疑う：confidenceがhighでないなら expected を優先
    # ※しきい値は作品によって調整。まずは >=7 が安全寄り
    if conf != "high" and abs(cut_num - expected) >= 7:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (jump_too_large)"]
        return expected, None, True, reasons, conf

    # page先頭rowは誤読が起きやすいので、もう少し強めに補正（必要なら）
    if row == 1 and conf != "high" and abs(cut_num - expected) >= 5:
        reasons = reasons + [f"sequence_repair: {cut_num}->{expected} (page_row1_bias)"]
        return expected, None, True, reasons, conf

    return cut_num, cut_num_str, is_new_cut, reasons, conf


def main():
    episode_dir = CFG["episode_dir"]
    episode_id = episode_id_from_dir(episode_dir)

    out_root = os.path.join(CFG["out_dir"], episode_id)
    cuts_out_dir = os.path.join(out_root, "cuts")
    ensure_dir(out_root)
    ensure_dir(cuts_out_dir)

    # Select pages
    all_pages = list_pages(episode_dir)
    if CFG["page_select"] == "all":
        pages = all_pages
    else:
        pages = [p for p in all_pages if CFG["start_page"] <= p <= CFG["end_page"]]

    if not pages:
        raise RuntimeError("No pages selected/found.")

    rows_per_page = CFG["rows_per_page"]
    cols_to_ocr = CFG["ocr_columns"]
    if "cut" not in cols_to_ocr:
        raise RuntimeError("CFG['ocr_columns'] must include 'cut'.")

    # Load model once
    model, processor = load_model()

    # OCR cache
    ocr_cache: Dict[str, Dict[str, Any]] = {}

    def ocr_path(path: str) -> Dict[str, Any]:
        if path in ocr_cache:
            return ocr_cache[path]
        img = load_image(path)
        res = ocr_one_image(model, processor, img)
        ocr_cache[path] = res
        return res

    # Store per-row OCR for final cut writing
    row_store: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # Build cut spans across pages
    cut_spans: List[CutSpan] = []
    current: Optional[CutSpan] = None
    previous_cut_num: Optional[int] = None
    previous_cut_num_str: Optional[str] = None

    for page in tqdm(pages, desc="Pages"):
        full_page_img = page_image_path(episode_dir, page)

        for row in tqdm(range(1, rows_per_page + 1), desc=f"Page {page} rows", leave=False):
            key = (page, row)
            row_store[key] = {"page_image": full_page_img, "cols": {}}

            # --- CUT OCR + decision
            cut_img_path = crop_path(episode_dir, "cut", page, row)
            if os.path.exists(cut_img_path):
                cut_ocr = ocr_path(cut_img_path)
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

            # Update previous cut string if we found a better zero-padded representation
            if cut_num is not None:
                if cut_num == previous_cut_num and cut_num_str:
                    # continuation row providing a better string form
                    if previous_cut_num_str is None or len(cut_num_str) > len(previous_cut_num_str):
                        previous_cut_num_str = cut_num_str
                elif is_new_cut:
                    previous_cut_num_str = cut_num_str

            # Handle new cut span
            if is_new_cut and cut_num is not None:
                # close previous span at previous row
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
                    end=(page, row),  # temporary
                    anchor=anchor,
                )
                previous_cut_num = cut_num

            else:
                # continuation (or no detected cut number)
                # If no current span yet, start an "unknown" span so we don't lose rows
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
                # previous_cut_num stays as-is

            # --- OCR other columns
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
                o = ocr_path(img_path)
                row_store[key]["cols"][col] = {"image": img_path, **o}

            # Always store picture path for Stage2 (if requested)
            if CFG["always_store_picture_path"] and "picture" not in row_store[key]["cols"]:
                pic_path = crop_path(episode_dir, "picture", page, row)
                row_store[key]["cols"]["picture"] = {"image": pic_path if os.path.exists(pic_path) else None}

            # If we have a current span, extend its end as we progress
            if current is not None:
                current.end = (page, row)

    # Close final span to the last processed row
    if current is not None:
        cut_spans.append(current)

    if not cut_spans:
        raise RuntimeError("No cut spans created. Check inputs and cut/ crops.")

    # Write cut files + index
    index: Dict[str, Any] = {
        "episode_id": episode_id,
        "episode_dir": episode_dir,
        "page_select": {k: CFG[k] for k in ["page_select", "start_page", "end_page"] if k in CFG},
        "rows_per_page": rows_per_page,
        "ocr_columns": cols_to_ocr,
        "cuts": [],
        "meta": {
            "model_id": CFG["model_id"],
            "created_at": now_iso_utc(),
            "gen": CFG["gen"],
        },
    }

    for span in cut_spans:
        start_page, start_row = span.start
        end_page, end_row = span.end

        # enumerate all (page,row) in span
        keys: List[Tuple[int, int]] = []
        for p in range(start_page, end_page + 1):
            r0 = start_row if p == start_page else 1
            r1 = end_row if p == end_page else rows_per_page
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
            "cut_str": span.cut_num_str,  # preserve zero padding when known
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

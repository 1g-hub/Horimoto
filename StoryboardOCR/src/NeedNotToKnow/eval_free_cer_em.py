# stage1_eval_free_cer_em.py
#
# Evaluate Stage1 OCR (FREE dataset) with GT in data/free/annotation.csv
# Metrics: CER (macro + micro), EM (exact match)
#
# Supports annotation CSV formats:
#  (A) LONG format:
#      cut,page,row,field,text
#  (B) WIDE format:
#      cut,page,row,action_memo,dialogue,cut,picture,time   (some subset)
#
# Prediction source:
#   /app/outputs/free/cuts/cutXXXX.stage1.json
#   rows[].cols[field].raw_text or rows[].cols[field].lines
#
# Output:
#   /app/outputs_eval/free_stage1_eval_cer_em.json
#   /app/outputs_eval/free_stage1_eval_samples.csv

import os
import re
import csv
import json
import glob
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


# ==================================================
# CONFIG
# ==================================================
APP_ROOT = "/app"
EPISODE_ID = "free"

ANNOT_CSV = f"{APP_ROOT}/data/free/annotation.csv"
STAGE1_CUTS_DIR = f"{APP_ROOT}/outputs/{EPISODE_ID}/cuts"

OUT_DIR = f"{APP_ROOT}/outputs/{EPISODE_ID}"
OUT_JSON = os.path.join(OUT_DIR, f"{EPISODE_ID}_stage1_eval_cer_em.json")
OUT_SAMPLES_CSV = os.path.join(OUT_DIR, f"{EPISODE_ID}_stage1_eval_samples.csv")

CFG = {
    # evaluate only these columns (if present in GT)
    "eval_columns": ["cut", "picture", "action_memo", "dialogue", "time"],

    # normalization: remove neutral placeholder symbols
    "neutral_symbols_removed": ["□", "△"],

    # whitespace normalize
    "collapse_spaces": True,

    # keep newlines or not (usually keep; set False if GT is single-line always)
    "keep_newlines": True,

    # if ref is empty:
    # - pred empty => CER=0, EM=1
    # - pred non-empty => CER=1, EM=0 (by definition here)
    "empty_ref_cer_if_pred_nonempty": 1.0,

    # if stage1 file missing => skip sample
    "skip_missing_stage1_file": True,

    # if a specific cell missing => skip sample
    "skip_missing_cell": True,
}
# ==================================================


CUT_STAGE1_RE = re.compile(r"cut(\d+)\.stage1\.json$", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_from_any(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def normalize_field_name(s: str) -> str:
    """
    Normalize field/column name variations to standard keys.
    """
    if not s:
        return ""
    t = str(s).strip().lower()
    t2 = re.sub(r"[\s_\-]+", "", t)

    alias = {
        "action": "action_memo",
        "actionmemo": "action_memo",
        "memo": "action_memo",
        "dialog": "dialogue",
        "dialogue": "dialogue",
        "serif": "dialogue",
        "speech": "dialogue",
        "time": "time",
        "cut": "cut",
        "picture": "picture",
        "img": "picture",
        "image": "picture",
    }
    if t2 in alias:
        return alias[t2]
    # if already exact
    if t in CFG["eval_columns"]:
        return t
    return t  # fallback


def normalize_text(s: Any) -> str:
    """
    Normalize both GT and prediction before CER/EM.
    - remove placeholder symbols □ △
    - normalize full-width spaces
    - optional collapse spaces
    - optional keep newlines
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")

    for sym in CFG["neutral_symbols_removed"]:
        s = s.replace(sym, "")

    # strip per-line
    if CFG["keep_newlines"]:
        lines = [ln.strip() for ln in s.split("\n")]
        # drop empty lines introduced by removals
        lines = [ln for ln in lines if ln != ""]
        s = "\n".join(lines)
    else:
        s = " ".join([ln.strip() for ln in s.split("\n") if ln.strip() != ""])

    if CFG["collapse_spaces"]:
        s = re.sub(r"[ \t]+", " ", s)

    return s.strip()


def join_stage1_cell_text(cell: Dict[str, Any]) -> str:
    """
    Stage1 cell format:
      {"raw_text": "...", "lines":[...], ...}
    Prefer raw_text; fallback to join lines.
    """
    if not isinstance(cell, dict):
        return ""
    raw = cell.get("raw_text")
    if isinstance(raw, str) and raw.strip() != "":
        return raw.strip()
    lines = cell.get("lines")
    if isinstance(lines, list) and lines:
        return "\n".join([str(x) for x in lines]).strip()
    return ""


def levenshtein_distance(a: str, b: str) -> int:
    """
    Space-efficient Levenshtein distance (edit distance).
    """
    if a == b:
        return 0
    if a is None:
        a = ""
    if b is None:
        b = ""
    a = str(a)
    b = str(b)

    # make b the shorter one for memory
    if len(a) < len(b):
        a, b = b, a

    # now len(a) >= len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def compute_cer_em(ref: str, pred: str) -> Tuple[float, int, int]:
    """
    Return (cer, edit_distance, ref_len).
    CER is defined as edit_distance / len(ref) when len(ref)>0.
    When len(ref)==0:
      - if pred empty: CER=0
      - else CER=CFG["empty_ref_cer_if_pred_nonempty"]
    """
    ref_len = len(ref)
    if ref_len == 0:
        if pred == "":
            return 0.0, 0, 0
        ed = len(pred)
        return float(CFG["empty_ref_cer_if_pred_nonempty"]), ed, 0

    ed = levenshtein_distance(ref, pred)
    cer = ed / max(ref_len, 1)
    return float(cer), int(ed), int(ref_len)


def find_header_key(header: List[str], candidates: List[str]) -> Optional[str]:
    """
    Find a header column matching candidates (case-insensitive).
    """
    low = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_annotation_csv(path: str) -> Tuple[Dict[Tuple[int, int, int, str], str], Dict[str, Any]]:
    """
    Load GT mapping:
      key = (cut, page, row, field)
      value = reference text
    Supports LONG or WIDE CSV.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"annotation.csv not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("annotation.csv has no header row")

        cut_k = find_header_key(header, ["cut", "cut_id", "cutid"])
        page_k = find_header_key(header, ["page", "page_id", "pageid"])
        row_k = find_header_key(header, ["row", "row_id", "rowid"])

        # LONG format keys
        field_k = find_header_key(header, ["field", "column", "col", "type", "name"])
        text_k = find_header_key(header, ["text", "gt", "ref", "reference", "label", "annotation", "answer"])

        # detect wide format: if any eval column exists as header
        header_norm = set([normalize_field_name(h) for h in header])
        has_any_eval_field = any([(c in header_norm) for c in CFG["eval_columns"]])

        gt_map: Dict[Tuple[int, int, int, str], str] = {}
        duplicates = 0
        n_rows = 0
        fmt = None

        for r in reader:
            n_rows += 1
            cut = parse_int_from_any(r.get(cut_k)) if cut_k else None
            page = parse_int_from_any(r.get(page_k)) if page_k else None
            row = parse_int_from_any(r.get(row_k)) if row_k else None

            if cut is None or page is None or row is None:
                # If your annotation uses a different ID style, extend here.
                continue

            # LONG format if (field_k and text_k) are present
            if field_k and text_k and r.get(field_k) is not None:
                fmt = fmt or "long"
                field = normalize_field_name(r.get(field_k, ""))
                if field not in CFG["eval_columns"]:
                    continue
                ref = normalize_text(r.get(text_k, ""))
                key = (cut, page, row, field)
                if key in gt_map:
                    duplicates += 1
                gt_map[key] = ref
                continue

            # Otherwise WIDE format if eval columns exist
            if has_any_eval_field:
                fmt = fmt or "wide"
                for col in header:
                    f_norm = normalize_field_name(col)
                    if f_norm not in CFG["eval_columns"]:
                        continue
                    ref = normalize_text(r.get(col, ""))
                    key = (cut, page, row, f_norm)
                    if key in gt_map:
                        duplicates += 1
                    gt_map[key] = ref

        meta = {
            "format_detected": fmt or "unknown",
            "n_csv_rows": n_rows,
            "n_gt_samples": len(gt_map),
            "n_duplicates_overwritten": duplicates,
            "header": header,
            "keys_used": {
                "cut": cut_k,
                "page": page_k,
                "row": row_k,
                "field": field_k,
                "text": text_k,
            },
        }
        if len(gt_map) == 0:
            raise ValueError(
                "No GT samples loaded from annotation.csv. "
                "Check header names (cut/page/row/field/text) or widen the detection logic."
            )
        return gt_map, meta


def build_stage1_pred_map(stage1_cuts_dir: str) -> Tuple[Dict[Tuple[int, int, int, str], str], Dict[str, Any]]:
    """
    Build prediction mapping from stage1 jsons:
      key=(cut,page,row,field) -> pred text
    """
    files = sorted(glob.glob(os.path.join(stage1_cuts_dir, "cut*.stage1.json")))
    if not files:
        raise FileNotFoundError(f"No stage1 cut jsons found: {stage1_cuts_dir}")

    pred_map: Dict[Tuple[int, int, int, str], str] = {}
    n_files = 0
    n_cells = 0

    for p in files:
        n_files += 1
        obj = read_json(p)
        cut = obj.get("cut")
        try:
            cut = int(cut)
        except Exception:
            cut = parse_int_from_any(cut)

        rows = obj.get("rows", []) or []
        for ro in rows:
            page = parse_int_from_any(ro.get("page"))
            row = parse_int_from_any(ro.get("row"))
            if cut is None or page is None or row is None:
                continue
            cols = ro.get("cols", {}) or {}
            for field in CFG["eval_columns"]:
                cell = cols.get(field, {}) or {}
                pred_raw = join_stage1_cell_text(cell)
                pred = normalize_text(pred_raw)
                key = (cut, page, row, field)
                pred_map[key] = pred
                n_cells += 1

    meta = {
        "n_stage1_files": n_files,
        "n_pred_cells_written": n_cells,
        "stage1_dir": stage1_cuts_dir,
    }
    return pred_map, meta


def init_acc() -> Dict[str, Any]:
    return {
        "n": 0,
        "sum_ref_len": 0,
        "sum_edit_distance": 0,
        "cer_macro_sum": 0.0,
        "exact_match_count": 0,
        "n_ref_empty": 0,
    }


def finalize_acc(acc: Dict[str, Any]) -> Dict[str, Any]:
    n = acc["n"]
    sum_ref_len = acc["sum_ref_len"]
    sum_edit = acc["sum_edit_distance"]
    cer_micro = (sum_edit / sum_ref_len) if sum_ref_len > 0 else None
    cer_macro = (acc["cer_macro_sum"] / n) if n > 0 else None
    em = (acc["exact_match_count"] / n) if n > 0 else None
    return {
        "n": n,
        "cer_macro": cer_macro,
        "cer_micro": cer_micro,
        "exact_match": em,
        "sum_ref_len": sum_ref_len,
        "sum_edit_distance": sum_edit,
        "n_ref_empty": acc["n_ref_empty"],
    }


def main():
    ensure_dir(OUT_DIR)

    # --- load GT ---
    gt_map, gt_meta = load_annotation_csv(ANNOT_CSV)

    # --- build predictions ---
    pred_map, pred_meta = build_stage1_pred_map(STAGE1_CUTS_DIR)

    # --- evaluate ---
    total_acc = init_acc()
    by_col: Dict[str, Dict[str, Any]] = {c: init_acc() for c in CFG["eval_columns"]}

    n_missing_pred = 0
    samples_out: List[Dict[str, Any]] = []

    # Evaluate ONLY keys present in GT (reference-driven)
    for (cut, page, row, field), ref in gt_map.items():
        pred = pred_map.get((cut, page, row, field))
        if pred is None:
            n_missing_pred += 1
            if CFG["skip_missing_cell"]:
                continue
            pred = ""

        # compute
        cer, ed, ref_len = compute_cer_em(ref, pred)
        em = 1 if ref == pred else 0

        # total
        total_acc["n"] += 1
        total_acc["cer_macro_sum"] += cer
        total_acc["sum_edit_distance"] += ed
        total_acc["sum_ref_len"] += ref_len
        total_acc["exact_match_count"] += em
        if ref_len == 0:
            total_acc["n_ref_empty"] += 1

        # by column
        if field in by_col:
            acc = by_col[field]
            acc["n"] += 1
            acc["cer_macro_sum"] += cer
            acc["sum_edit_distance"] += ed
            acc["sum_ref_len"] += ref_len
            acc["exact_match_count"] += em
            if ref_len == 0:
                acc["n_ref_empty"] += 1

        samples_out.append({
            "cut": cut,
            "page": page,
            "row": row,
            "field": field,
            "ref": ref,
            "pred": pred,
            "edit_distance": ed,
            "ref_len": ref_len,
            "cer": cer,
            "exact_match": em,
        })

    # finalize
    result = {
        "meta": {
            "created_at": now_iso(),
            "episode_id": EPISODE_ID,
            "annotation_csv": ANNOT_CSV,
            "stage1_cuts_dir": STAGE1_CUTS_DIR,
            "eval_columns": CFG["eval_columns"],
            "neutral_symbols_removed": CFG["neutral_symbols_removed"],
            "normalization": {
                "collapse_spaces": CFG["collapse_spaces"],
                "keep_newlines": CFG["keep_newlines"],
            },
            "gt_meta": gt_meta,
            "pred_meta": pred_meta,
        },
        "counts": {
            "gt_samples": len(gt_map),
            "evaluated_samples": total_acc["n"],
            "missing_pred_for_gt": n_missing_pred,
        },
        "total": finalize_acc(total_acc),
        "by_column": {c: finalize_acc(by_col[c]) for c in CFG["eval_columns"]},
    }

    # write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {OUT_JSON}")

    # write per-sample CSV
    with open(OUT_SAMPLES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cut", "page", "row", "field", "ref", "pred", "edit_distance", "ref_len", "cer", "exact_match"]
        )
        writer.writeheader()
        for r in samples_out:
            writer.writerow(r)
    print(f"[OK] wrote {OUT_SAMPLES_CSV}")

    # print short summary
    print("\n=== SUMMARY ===")
    print("TOTAL:", result["total"])
    print("BY COLUMN:")
    for c, v in result["by_column"].items():
        print(f"  - {c}: {v}")


if __name__ == "__main__":
    main()

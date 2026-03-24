# make_ocr_annotation_csv.py

import os
import re
import json
import glob
import csv
from typing import Dict, Any, List, Optional

# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    "stage1_cuts_dir": "outputs/episode01/cuts",
    "stage2_cuts_dir": "outputs_stage2/episode01/cuts",
    "out_csv": "outputs/episode01/annotation_table.csv",
    "export_columns": ["cut", "picture", "action_memo", "dialogue", "time"],

    # NEW: final_gt の初期値方針
    # True: stage2_corrected があればそれを入れる。無ければ stage1_ocr。
    "prefill_final_gt": True,
}
# =========================

HEADERS = [
    "cut",
    "page",
    "row",
    "column",
    "stage1_ocr",
    "stage2_corrected",
    "final_gt",
    "error_type",
    "note",
]

CUT_RE = re.compile(r"cut(\d+)\.stage[12]\.json$")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_cut_files(dir_path: str, stage: int) -> List[str]:
    pat = os.path.join(dir_path, f"cut*.stage{stage}.json")
    return sorted(glob.glob(pat))


def parse_cut_number(path: str) -> Optional[int]:
    m = CUT_RE.search(path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def build_stage2_map(stage2_dir: str) -> Dict[int, Dict[str, Any]]:
    m = {}
    for p in list_cut_files(stage2_dir, stage=2):
        cut_num = parse_cut_number(p)
        if cut_num is not None:
            m[cut_num] = read_json(p)
    return m


def stage2_corrected_for_column(stage2_obj: Optional[Dict[str, Any]], col: str) -> str:
    if not stage2_obj:
        return ""

    ocr_norm = stage2_obj.get("ocr_norm", {}) or {}
    ocr_raw = stage2_obj.get("ocr_raw", {}) or {}

    if col == "action_memo":
        return str(ocr_norm.get("action_memo", "") or "")
    if col == "dialogue":
        return str(ocr_norm.get("dialogue", "") or "")
    if col == "time":
        return str(
            ocr_norm.get("time_norm")
            or ocr_norm.get("time")
            or ocr_raw.get("time_raw")
            or ""
        )

    # cut/picture は Stage2 側に補正結果が無いので空
    return ""


def safe_stage1_text(cols: Dict[str, Any], col: str) -> str:
    cell = cols.get(col, {}) or {}
    return str(cell.get("raw_text", "") or "")


def make_rows(stage1_obj: Dict[str, Any], stage2_obj: Optional[Dict[str, Any]]) -> List[List[str]]:
    rows_out = []
    cut_num = stage1_obj.get("cut", "")

    for r in stage1_obj.get("rows", []):
        page = r.get("page", "")
        row = r.get("row", "")
        cols = r.get("cols", {}) or {}

        for col in CFG["export_columns"]:
            stage1_ocr = safe_stage1_text(cols, col)
            stage2_corr = stage2_corrected_for_column(stage2_obj, col)

            # NEW: final_gt を事前に埋める
            if CFG["prefill_final_gt"]:
                final_gt = stage1_ocr
            else:
                final_gt = ""

            rows_out.append([
                str(cut_num),
                str(page),
                str(row),
                col,
                stage1_ocr,
                stage2_corr,
                final_gt,
                "",  # error_type
                "",  # note
            ])

    return rows_out


def main():
    stage1_files = list_cut_files(CFG["stage1_cuts_dir"], stage=1)
    if not stage1_files:
        raise FileNotFoundError(f"No stage1 files found: {CFG['stage1_cuts_dir']}")

    stage2_map = build_stage2_map(CFG["stage2_cuts_dir"])

    os.makedirs(os.path.dirname(CFG["out_csv"]) or ".", exist_ok=True)

    # UTF-8 with BOM → Excelで文字化け防止
    with open(CFG["out_csv"], "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

        for p in stage1_files:
            cut_num = parse_cut_number(p)
            stage1_obj = read_json(p)
            stage2_obj = stage2_map.get(cut_num)

            rows = make_rows(stage1_obj, stage2_obj)
            for r in rows:
                writer.writerow(r)

    print(f"[OK] wrote {CFG['out_csv']}")


if __name__ == "__main__":
    main()

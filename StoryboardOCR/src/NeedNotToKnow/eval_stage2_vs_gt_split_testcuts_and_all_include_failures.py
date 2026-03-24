# eval_stage2_vs_gt_split_testcuts_and_all_include_failures.py
# ------------------------------------------------------------
# Evaluate Stage2 (CLM correction) outputs against GT CSV.
#
# Fix:
# - Previously, samples were SKIPPED if stage2 cut file/cell was missing.
# - Now, if stage2 is missing, we TREAT stage2 output as stage1 output (no change),
#   and INCLUDE those samples in evaluation.
#
# Stage1 source (fallback & stage1 metrics):
#   outputs/episode01/cuts/cutXXXX.stage1.json
#
# Stage2 source:
#   outputs_stage2_clm/episode01/cuts_split/cutXXXX.stage2.split.{nojp|jp}.json
#
# GT CSV header (your format):
#   cut,page,row,column,stage1_ocr,stage2_corrected,final_gt,error_type,note
# ------------------------------------------------------------

import os
import re
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


# =========================
# CONFIG (EDIT HERE)
# =========================
APP_ROOT = "/app"

CFG = {
    # episode
    "episode_id": "episode01",

    # training split (has test_cut_ids)
    "split_by_cut_json": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train50per/split_by_cut.json",

    # GT table (your header)
    "csv_path": "data/episode01/annotation_table.csv",

    # Stage1 outputs (IMPORTANT: used for stage1 metrics and stage2 fallback)
    "stage1_cuts_dir": "outputs/episode01/cuts",

    # Stage2 outputs
    "stage2_split_dir": "outputs_stage2_clm/episode01/cuts_split",

    # Evaluate which Stage2 variant
    "stage2_variant": "both",  # "nojp" | "jp" | "both"

    # Evaluate columns
    "eval_columns": {"cut", "picture", "action_memo", "dialogue", "time"},

    # Normalization: remove these symbols before CER/ExactMatch
    "neutral_symbols": ["□", "△"],

    # Output
    "out_dir": "outputs_stage2_clm/episode01/eval_stage2_vs_gt_include_failures_pic_act_dia",
    "write_samples_jsonl": True,   # dump per-sample results
    "max_samples_jsonl": None,     # or int
}
# =========================


# -------------------------
# Path helpers
# -------------------------
def resolve_path(p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(APP_ROOT, p)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# -------------------------
# Normalization + edit distance
# -------------------------
def normalize_for_eval(s: str, neutral_symbols: List[str]) -> str:
    """
    - collapse whitespace
    - remove neutral symbols
    - keep other characters as-is (incl arrows)
    """
    s = s or ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for sym in neutral_symbols:
        s = s.replace(sym, "")
    s = " ".join(s.split())
    return s

def levenshtein(seq1: List[str], seq2: List[str]) -> int:
    if seq1 == seq2:
        return 0
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    prev = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, 1):
        cur = [i]
        for j, b in enumerate(seq2, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if a == b else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def cer_and_dist(hyp: str, ref: str, neutral_symbols: List[str]) -> Tuple[float, int, int]:
    """
    Returns:
      cer (uncapped), dist, ref_len
    """
    hyp2 = normalize_for_eval(hyp, neutral_symbols)
    ref2 = normalize_for_eval(ref, neutral_symbols)
    h = list(hyp2)
    r = list(ref2)
    if len(r) == 0:
        # if ref empty: define CER = 0 if hyp also empty else 1
        return (0.0 if len(h) == 0 else 1.0), (0 if len(h) == 0 else len(h)), 0
    d = levenshtein(h, r)
    return d / len(r), d, len(r)

def exact_match(hyp: str, ref: str, neutral_symbols: List[str]) -> int:
    return int(normalize_for_eval(hyp, neutral_symbols) == normalize_for_eval(ref, neutral_symbols))


# -------------------------
# Aggregators
# -------------------------
class Agg:
    def __init__(self):
        self.n = 0
        self.sum_cer = 0.0              # macro over samples
        self.sum_em = 0
        self.sum_dist = 0               # for micro CER
        self.sum_ref_len = 0
        self.sum_characc = 0.0          # 1 - CER (macro)
        self.sum_characc_clipped = 0.0  # clipped to [0,1]

    def add(self, hyp: str, ref: str, neutral_symbols: List[str]):
        c, d, rlen = cer_and_dist(hyp, ref, neutral_symbols)
        em = exact_match(hyp, ref, neutral_symbols)

        self.n += 1
        self.sum_cer += float(c)
        self.sum_em += int(em)
        self.sum_dist += int(d)
        self.sum_ref_len += int(rlen)

        characc = 1.0 - float(c)
        self.sum_characc += characc
        c_clip = min(1.0, max(0.0, float(c)))
        self.sum_characc_clipped += (1.0 - c_clip)

    def finalize(self) -> Dict[str, Any]:
        n = max(self.n, 1)
        macro_cer = self.sum_cer / n
        macro_em = self.sum_em / n
        macro_characc = self.sum_characc / n
        macro_characc_clip = self.sum_characc_clipped / n

        micro_cer = (self.sum_dist / self.sum_ref_len) if self.sum_ref_len > 0 else 0.0
        micro_characc = 1.0 - micro_cer
        micro_characc_clip = 1.0 - min(1.0, max(0.0, micro_cer))

        return {
            "n": self.n,
            "cer_macro_over_samples": macro_cer,
            "cer_micro": micro_cer,
            "exact_match": macro_em,
            "char_accuracy_macro": macro_characc,
            "char_accuracy_macro_clipped_0_1": macro_characc_clip,
            "char_accuracy_micro": micro_characc,
            "char_accuracy_micro_clipped_0_1": micro_characc_clip,
            "sum_ref_len": self.sum_ref_len,
            "sum_edit_distance": self.sum_dist,
        }


def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def macro_by_cut(aggs_by_cut: Dict[int, Agg]) -> Dict[str, Any]:
    """
    Macro over cuts, using per-cut MICRO CER and per-cut ExactMatch mean.
    """
    cuts = sorted(aggs_by_cut.keys())
    if not cuts:
        return {"n_cuts": 0}

    micro_cers = []
    ems = []
    for c in cuts:
        a = aggs_by_cut[c]
        f = a.finalize()
        micro_cers.append(float(f["cer_micro"]))
        ems.append(float(f["exact_match"]))

    return {
        "n_cuts": len(cuts),
        "macro_over_cuts__cer_micro_mean": mean(micro_cers),
        "macro_over_cuts__exact_match_mean": mean(ems),
    }


# -------------------------
# Stage1/Stage2 loaders
# -------------------------
S2_SPLIT_RE = re.compile(r"^cut(\d+)\.stage2\.split\.(nojp|jp)\.json$")
S1_RE = re.compile(r"^cut(\d+)\.stage1\.json$")

def build_stage2_split_path_map(stage2_split_dir: str) -> Dict[str, Dict[int, str]]:
    """
    Returns: paths[variant][cut_int] = filepath
    """
    paths: Dict[str, Dict[int, str]] = {"nojp": {}, "jp": {}}
    stage2_split_dir_abs = resolve_path(stage2_split_dir)
    if not os.path.isdir(stage2_split_dir_abs):
        return paths

    for fn in os.listdir(stage2_split_dir_abs):
        m = S2_SPLIT_RE.match(fn)
        if not m:
            continue
        cut_i = int(m.group(1))
        variant = m.group(2)
        paths[variant][cut_i] = os.path.join(stage2_split_dir_abs, fn)
    return paths

def build_stage1_path_map(stage1_cuts_dir: str) -> Dict[int, str]:
    """
    stage1: outputs/<episode>/cuts/cutXXXX.stage1.json
    """
    mp: Dict[int, str] = {}
    stage1_cuts_dir_abs = resolve_path(stage1_cuts_dir)
    if not os.path.isdir(stage1_cuts_dir_abs):
        return mp

    for fn in os.listdir(stage1_cuts_dir_abs):
        m = S1_RE.match(fn)
        if not m:
            continue
        cut_i = int(m.group(1))
        mp[cut_i] = os.path.join(stage1_cuts_dir_abs, fn)
    return mp

def _stage1_cell_text(cell: Dict[str, Any]) -> str:
    """
    stage1 cell has raw_text and/or lines.
    """
    if not isinstance(cell, dict):
        return ""
    raw = cell.get("raw_text")
    if isinstance(raw, str):
        return raw
    lines = cell.get("lines")
    if isinstance(lines, list):
        return "\n".join([str(x) for x in lines])
    return ""

def load_stage1_cells(path: str) -> Dict[Tuple[int, int, str], str]:
    """
    Build map:
      (page, row, column) -> raw_text
    """
    obj = read_json(path)
    out: Dict[Tuple[int, int, str], str] = {}

    rows = obj.get("rows", []) or []
    for r in rows:
        try:
            page = int(r.get("page"))
            row = int(r.get("row"))
        except Exception:
            continue
        cols = r.get("cols", {}) or {}
        for col, cell in cols.items():
            txt = _stage1_cell_text(cell or {})
            if not isinstance(txt, str):
                txt = str(txt)
            out[(page, row, str(col))] = txt
    return out

def load_stage2_split_cells(path: str) -> Dict[Tuple[int, int, str], Dict[str, str]]:
    """
    Build map:
      (page, row, column) -> {"raw_text": ..., "corrected_text": ...}
    """
    obj = read_json(path)
    out: Dict[Tuple[int, int, str], Dict[str, str]] = {}

    rows = obj.get("rows", []) or []
    for r in rows:
        try:
            page = int(r.get("page"))
            row = int(r.get("row"))
        except Exception:
            continue
        cols = r.get("cols", {}) or {}
        for col, cell in cols.items():
            cell = cell or {}
            raw = cell.get("raw_text", "")
            cor = cell.get("corrected_text", raw)
            if not isinstance(raw, str):
                raw = str(raw)
            if not isinstance(cor, str):
                cor = str(cor)
            out[(page, row, str(col))] = {"raw_text": raw, "corrected_text": cor}

    return out


# -------------------------
# CSV samples
# -------------------------
@dataclass
class Sample:
    cut: int
    page: int
    row: int
    column: str
    ref: str


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == "none":
            return None
        return int(s)
    except Exception:
        return None


def load_eval_samples_from_csv(csv_path: str, eval_columns: set) -> List[Sample]:
    """
    Your CSV header:
      cut,page,row,column,stage1_ocr,stage2_corrected,final_gt,error_type,note
    We only need cut/page/row/column/final_gt for evaluation.
    """
    rows = read_csv_rows(resolve_path(csv_path))
    out: List[Sample] = []
    for r in rows:
        col = str(r.get("column", "") or "").strip()
        if eval_columns and col not in eval_columns:
            continue

        cut_i = safe_int(r.get("cut"))
        page_i = safe_int(r.get("page"))
        row_i = safe_int(r.get("row"))
        if cut_i is None or page_i is None or row_i is None:
            continue

        ref = (r.get("final_gt") or "").strip()
        if not ref:
            continue

        out.append(Sample(cut=cut_i, page=page_i, row=row_i, column=col, ref=ref))
    return out


def compute_delta(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
    """
    delta = b - a for selected numeric keys if present
    """
    out = {}
    for k in keys:
        if k in a and k in b:
            try:
                out[k] = float(b[k]) - float(a[k])
            except Exception:
                pass
    return out


# -------------------------
# Evaluation core (IMPORTANT FIX HERE)
# -------------------------
def evaluate_subset(
    *,
    samples: List[Sample],
    stage1_path_map: Dict[int, str],
    stage2_paths_for_variant: Dict[int, str],
    neutral_symbols: List[str],
    subset_name: str,
    write_samples_jsonl: bool,
    out_jsonl_path: Optional[str],
) -> Dict[str, Any]:
    """
    Evaluate Stage1(raw) and Stage2(corrected) for given samples.
    If stage2 file/cell missing -> stage2 hyp = stage1 hyp (fallback).
    """
    # caches
    s1_cut_cache: Dict[int, Dict[Tuple[int, int, str], str]] = {}
    s2_cut_cache: Dict[int, Dict[Tuple[int, int, str], Dict[str, str]]] = {}

    # totals
    s1_total = Agg()
    s2_total = Agg()
    s1_by_col: Dict[str, Agg] = {}
    s2_by_col: Dict[str, Agg] = {}
    s1_by_cut: Dict[int, Agg] = {}
    s2_by_cut: Dict[int, Agg] = {}

    # counts
    missing_stage1_file = 0
    missing_stage1_cell = 0
    missing_stage2_file = 0
    missing_stage2_cell = 0
    stage2_fallback_used = 0

    jsonl_f = None
    dumped = 0
    if write_samples_jsonl and out_jsonl_path:
        os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
        jsonl_f = open(out_jsonl_path, "w", encoding="utf-8")

    try:
        for sp in samples:
            cut = sp.cut

            # ---- stage1 (must exist to evaluate) ----
            s1_path = stage1_path_map.get(cut)
            if not s1_path or (not os.path.isfile(s1_path)):
                missing_stage1_file += 1
                continue

            if cut not in s1_cut_cache:
                s1_cut_cache[cut] = load_stage1_cells(s1_path)

            s1_cells = s1_cut_cache[cut]
            key = (sp.page, sp.row, sp.column)
            hyp_s1 = s1_cells.get(key)
            if hyp_s1 is None:
                missing_stage1_cell += 1
                continue

            # ---- stage2 (optional; fallback to stage1 if missing) ----
            s2_path = stage2_paths_for_variant.get(cut)
            hyp_s2 = None
            used_fallback = False
            fallback_reason = None

            if not s2_path or (not os.path.isfile(s2_path)):
                missing_stage2_file += 1
                hyp_s2 = hyp_s1
                used_fallback = True
                fallback_reason = "missing_stage2_file"
            else:
                if cut not in s2_cut_cache:
                    s2_cut_cache[cut] = load_stage2_split_cells(s2_path)
                s2_cells = s2_cut_cache[cut]
                cell2 = s2_cells.get(key)
                if cell2 is None:
                    missing_stage2_cell += 1
                    hyp_s2 = hyp_s1
                    used_fallback = True
                    fallback_reason = "missing_stage2_cell"
                else:
                    hyp_s2 = cell2.get("corrected_text", hyp_s1)
                    if not isinstance(hyp_s2, str):
                        hyp_s2 = str(hyp_s2)

            if used_fallback:
                stage2_fallback_used += 1

            ref = sp.ref

            # add to totals
            s1_total.add(hyp_s1, ref, neutral_symbols)
            s2_total.add(hyp_s2, ref, neutral_symbols)

            # by column
            if sp.column not in s1_by_col:
                s1_by_col[sp.column] = Agg()
                s2_by_col[sp.column] = Agg()
            s1_by_col[sp.column].add(hyp_s1, ref, neutral_symbols)
            s2_by_col[sp.column].add(hyp_s2, ref, neutral_symbols)

            # by cut
            if cut not in s1_by_cut:
                s1_by_cut[cut] = Agg()
                s2_by_cut[cut] = Agg()
            s1_by_cut[cut].add(hyp_s1, ref, neutral_symbols)
            s2_by_cut[cut].add(hyp_s2, ref, neutral_symbols)

            # dump jsonl
            if jsonl_f is not None:
                if CFG["max_samples_jsonl"] is None or dumped < int(CFG["max_samples_jsonl"]):
                    cer1, _, _ = cer_and_dist(hyp_s1, ref, neutral_symbols)
                    cer2, _, _ = cer_and_dist(hyp_s2, ref, neutral_symbols)
                    rec = {
                        "subset": subset_name,
                        "cut": cut,
                        "page": sp.page,
                        "row": sp.row,
                        "column": sp.column,
                        "ref": ref,
                        "hyp_stage1_raw": hyp_s1,
                        "hyp_stage2_corrected": hyp_s2,
                        "used_stage2_fallback_to_stage1": used_fallback,
                        "fallback_reason": fallback_reason,
                        "stage1_path": s1_path,
                        "stage2_path": s2_path if s2_path else None,
                        "cer_stage1": float(cer1),
                        "cer_stage2": float(cer2),
                        "delta_cer_stage2_minus_stage1": float(cer2 - cer1),
                        "exact_stage1": int(exact_match(hyp_s1, ref, neutral_symbols)),
                        "exact_stage2": int(exact_match(hyp_s2, ref, neutral_symbols)),
                    }
                    jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    dumped += 1

        # finalize
        s1_total_f = s1_total.finalize()
        s2_total_f = s2_total.finalize()

        s1_by_col_f = {c: a.finalize() for c, a in sorted(s1_by_col.items())}
        s2_by_col_f = {c: a.finalize() for c, a in sorted(s2_by_col.items())}

        s1_macro_cut = macro_by_cut(s1_by_cut)
        s2_macro_cut = macro_by_cut(s2_by_cut)

        delta_total = compute_delta(
            s1_total_f, s2_total_f,
            keys=["cer_macro_over_samples", "cer_micro", "exact_match",
                  "char_accuracy_macro", "char_accuracy_micro"]
        )

        return {
            "subset": subset_name,
            "counts": {
                "samples_input": len(samples),
                "samples_evaluated": s1_total_f["n"],
                "missing_stage1_file": missing_stage1_file,
                "missing_cell_in_stage1": missing_stage1_cell,
                "missing_stage2_file": missing_stage2_file,
                "missing_cell_in_stage2": missing_stage2_cell,
                "stage2_fallback_used_to_stage1": stage2_fallback_used,
            },
            "total": {
                "stage1_raw": s1_total_f,
                "stage2_corrected": s2_total_f,
                "delta_stage2_minus_stage1": delta_total,
            },
            "by_column": {
                "stage1_raw": s1_by_col_f,
                "stage2_corrected": s2_by_col_f,
            },
            "macro_by_cut": {
                "stage1_raw": s1_macro_cut,
                "stage2_corrected": s2_macro_cut,
            }
        }

    finally:
        if jsonl_f is not None:
            jsonl_f.close()


# -------------------------
# Main per variant
# -------------------------
def main_one_variant(variant: str):
    out_dir = resolve_path(CFG["out_dir"])
    ensure_dir(out_dir)

    # Load split_by_cut.json -> test_cut_ids
    split = read_json(resolve_path(CFG["split_by_cut_json"]))
    test_cut_ids_str = (split.get("split_info", {}) or {}).get("test_cut_ids", []) or []
    test_cut_ids = set()
    for s in test_cut_ids_str:
        try:
            test_cut_ids.add(int(str(s)))
        except Exception:
            pass

    # Load CSV -> samples
    all_samples = load_eval_samples_from_csv(CFG["csv_path"], CFG["eval_columns"])
    test_samples = [sp for sp in all_samples if sp.cut in test_cut_ids]

    # Stage1 path map
    stage1_path_map = build_stage1_path_map(CFG["stage1_cuts_dir"])

    # Stage2 file map
    paths = build_stage2_split_path_map(CFG["stage2_split_dir"])
    stage2_paths_for_variant = paths.get(variant, {})

    # Evaluate
    out_jsonl_all = os.path.join(out_dir, f"samples_all.{variant}.jsonl") if CFG["write_samples_jsonl"] else None
    out_jsonl_test = os.path.join(out_dir, f"samples_testcuts.{variant}.jsonl") if CFG["write_samples_jsonl"] else None

    res_test = evaluate_subset(
        samples=test_samples,
        stage1_path_map=stage1_path_map,
        stage2_paths_for_variant=stage2_paths_for_variant,
        neutral_symbols=CFG["neutral_symbols"],
        subset_name="train_test_cut_ids_only",
        write_samples_jsonl=CFG["write_samples_jsonl"],
        out_jsonl_path=out_jsonl_test,
    )

    res_all = evaluate_subset(
        samples=all_samples,
        stage1_path_map=stage1_path_map,
        stage2_paths_for_variant=stage2_paths_for_variant,
        neutral_symbols=CFG["neutral_symbols"],
        subset_name="all_cuts",
        write_samples_jsonl=CFG["write_samples_jsonl"],
        out_jsonl_path=out_jsonl_all,
    )

    out = {
        "meta": {
            "created_at": now_iso_utc(),
            "episode_id": CFG["episode_id"],
            "stage2_variant": variant,
            "split_by_cut_json": CFG["split_by_cut_json"],
            "csv_path": CFG["csv_path"],
            "stage1_cuts_dir": CFG["stage1_cuts_dir"],
            "stage2_split_dir": CFG["stage2_split_dir"],
            "eval_columns": sorted(list(CFG["eval_columns"])),
            "neutral_symbols_removed": CFG["neutral_symbols"],
            "note": "Stage2 missing cut/cell is evaluated by treating stage2 output == stage1 output (fallback).",
        },
        "train_split": {
            "n_test_cut_ids": len(test_cut_ids),
        },
        "results": {
            "test_cut_ids_only": res_test,
            "all_cuts": res_all,
        }
    }

    out_path = os.path.join(out_dir, f"eval_stage2_vs_gt_split.include_failures.{variant}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Print short summary
    def _p(label: str, obj: Dict[str, Any]):
        t = obj["total"]
        s1 = t["stage1_raw"]
        s2 = t["stage2_corrected"]
        d = t["delta_stage2_minus_stage1"]
        c = obj["counts"]
        print(f"\n=== {label} ({variant}) ===")
        print("samples_input:", c["samples_input"], " evaluated:", c["samples_evaluated"])
        print("missing_stage2_file:", c["missing_stage2_file"], " missing_stage2_cell:", c["missing_cell_in_stage2"],
              " fallback_used:", c["stage2_fallback_used_to_stage1"])
        print("[Stage1 raw]  CER_micro=", s1["cer_micro"], " CER_macro=", s1["cer_macro_over_samples"], " EM=", s1["exact_match"])
        print("[Stage2 corr] CER_micro=", s2["cer_micro"], " CER_macro=", s2["cer_macro_over_samples"], " EM=", s2["exact_match"])
        print("[Delta S2-S1] ", d)

    _p("TEST_CUT_IDS_ONLY", res_test)
    _p("ALL_CUTS", res_all)

    print(f"\n[OK] wrote {out_path}")
    if CFG["write_samples_jsonl"]:
        print(f"[OK] wrote JSONL: {out_jsonl_test}")
        print(f"[OK] wrote JSONL: {out_jsonl_all}")


def main():
    v = CFG["stage2_variant"]
    if v == "both":
        main_one_variant("nojp")
        main_one_variant("jp")
    else:
        if v not in ("nojp", "jp"):
            raise ValueError("CFG['stage2_variant'] must be 'nojp', 'jp', or 'both'")
        main_one_variant(v)


if __name__ == "__main__":
    main()

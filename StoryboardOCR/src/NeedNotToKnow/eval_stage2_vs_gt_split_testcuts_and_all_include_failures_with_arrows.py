# eval_stage2_vs_gt_split_testcuts_and_all_include_failures_with_arrows.py
# ------------------------------------------------------------
# Evaluate Stage2 (CLM correction) outputs against GT CSV.
#
# Fix (include failures):
# - If Stage2 cut file/cell is missing, treat Stage2 output == Stage1 output
#   and INCLUDE the sample in evaluation.
#
# Add (arrow-aware metrics):
# - Arrow Presence Recall
# - Arrow Count Exact
# - Direction Precision / Recall / F1 (multiset match by arrow direction)
# - Arrow Seq Similarity (edit-distance similarity on arrow-only sequences)
#
# Arrow metrics are aligned with the definitions in thesis section 4.4.2.
# ------------------------------------------------------------

import os
import re
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


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

    # Stage1 outputs (used for stage1 metrics and stage2 fallback)
    "stage1_cuts_dir": "outputs_LoRA_dia/episode01/cuts",

    # Stage2 outputs
    "stage2_split_dir": "outputs_stage2_clm/episode01/cuts_split",

    # Evaluate which Stage2 variant
    "stage2_variant": "both",  # "nojp" | "jp" | "both"

    # Evaluate columns
    "eval_columns": {"cut", "picture", "action_memo", "dialogue", "time"},

    # Normalization: remove these symbols before CER/ExactMatch
    "neutral_symbols": ["□", "△"],

    # Arrow settings
    "arrow_set": ["→", "←", "↑", "↓"],

    # Arrow Seq Similarity averaging:
    # - "ref_has_arrow": average only where reference contains >=1 arrow (S+)
    # - "ref_or_hyp_has_arrow": average where either ref or hyp has arrows (avoid empty-empty=1)
    "arrow_seq_similarity_scope": "ref_has_arrow",

    # Output
    "out_dir": "outputs_stage2_clm/episode01/eval_stage2_vs_gt_include_failures_with_arrows_pic_dia",
    "write_samples_jsonl": True,
    "max_samples_jsonl": None,  # or int
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
# Normalization + edit distance (CER/EM)
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
        return (0.0 if len(h) == 0 else 1.0), (0 if len(h) == 0 else len(h)), 0
    d = levenshtein(h, r)
    return d / len(r), d, len(r)

def exact_match(hyp: str, ref: str, neutral_symbols: List[str]) -> int:
    return int(normalize_for_eval(hyp, neutral_symbols) == normalize_for_eval(ref, neutral_symbols))


# -------------------------
# Arrow metrics helpers
# -------------------------
def extract_arrows(s: str, arrow_set: List[str]) -> List[str]:
    """
    Extract arrows in appearance order.
    """
    if not s:
        return []
    aset = set(arrow_set)
    return [ch for ch in s if ch in aset]

def arrow_seq_similarity(hyp_ar: List[str], ref_ar: List[str]) -> float:
    """
    1 - D(h, r) / max(|r|, |h|, 1)
    """
    d = levenshtein(hyp_ar, ref_ar)
    denom = max(len(ref_ar), len(hyp_ar), 1)
    return 1.0 - (d / float(denom))

def safe_mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / float(len(xs))


class ArrowAgg:
    """
    Aggregate arrow-aware metrics over samples.

    Definitions (thesis 4.4.2 compatible):
      - ArrowPresenceRecall: among samples where ref has >=1 arrow, fraction where hyp has >=1 arrow
      - ArrowCountExact: among samples where ref has >=1 arrow, fraction where len(hyp_ar)==len(ref_ar)
      - Direction Precision/Recall/F1: multiset match by direction across all samples
        TP = sum_i sum_a min(c_ref[i,a], c_hyp[i,a])
        Pred = sum_i sum_a c_hyp[i,a]
        Ref  = sum_i sum_a c_ref[i,a]
      - ArrowSeqSimilarity: average of per-sample similarity on arrow-only sequences
    """
    def __init__(self, arrow_set: List[str], seq_scope: str = "ref_has_arrow"):
        self.arrow_set = list(arrow_set)
        self.aset = set(arrow_set)
        self.seq_scope = seq_scope

        self.n_samples = 0

        # S+ (ref has >=1 arrow)
        self.n_ref_has_arrow = 0
        self.n_ref_has_arrow_and_hyp_has_arrow = 0
        self.n_ref_has_arrow_and_count_exact = 0

        # Direction totals
        self.total_ref_arrows = 0
        self.total_hyp_arrows = 0
        self.total_tp = 0

        # Sequence similarity
        self.n_seq_eval = 0
        self.sum_seq_sim = 0.0

    def add(self, hyp: str, ref: str):
        self.n_samples += 1

        hyp_ar = extract_arrows(hyp, self.arrow_set)
        ref_ar = extract_arrows(ref, self.arrow_set)

        # S+ logic (presence/count exact)
        if len(ref_ar) > 0:
            self.n_ref_has_arrow += 1
            if len(hyp_ar) > 0:
                self.n_ref_has_arrow_and_hyp_has_arrow += 1
            if len(hyp_ar) == len(ref_ar):
                self.n_ref_has_arrow_and_count_exact += 1

        # Direction totals (include all samples; samples with ref empty contribute FP if hyp has arrows)
        cref = Counter(ref_ar)
        chyp = Counter(hyp_ar)
        tp = 0
        for a in self.arrow_set:
            tp += min(int(cref.get(a, 0)), int(chyp.get(a, 0)))

        self.total_tp += int(tp)
        self.total_ref_arrows += int(len(ref_ar))
        self.total_hyp_arrows += int(len(hyp_ar))

        # Seq similarity scope
        do_seq = False
        if self.seq_scope == "ref_has_arrow":
            do_seq = (len(ref_ar) > 0)
        elif self.seq_scope == "ref_or_hyp_has_arrow":
            do_seq = (len(ref_ar) > 0) or (len(hyp_ar) > 0)
        else:
            # fallback
            do_seq = (len(ref_ar) > 0)

        if do_seq:
            sim = arrow_seq_similarity(hyp_ar, ref_ar)
            self.sum_seq_sim += float(sim)
            self.n_seq_eval += 1

    def finalize(self) -> Dict[str, Any]:
        # Presence recall / count exact over S+
        if self.n_ref_has_arrow > 0:
            presence_recall = self.n_ref_has_arrow_and_hyp_has_arrow / float(self.n_ref_has_arrow)
            count_exact = self.n_ref_has_arrow_and_count_exact / float(self.n_ref_has_arrow)
        else:
            presence_recall = None
            count_exact = None

        # Direction precision/recall/f1 (micro)
        prec = (self.total_tp / float(self.total_hyp_arrows)) if self.total_hyp_arrows > 0 else 0.0
        rec = (self.total_tp / float(self.total_ref_arrows)) if self.total_ref_arrows > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Seq similarity (macro over evaluated seq samples)
        seq_sim = (self.sum_seq_sim / float(self.n_seq_eval)) if self.n_seq_eval > 0 else None

        return {
            "n_samples": self.n_samples,
            "n_ref_has_arrow": self.n_ref_has_arrow,
            "arrow_presence_recall": presence_recall,
            "arrow_count_exact": count_exact,
            "direction_precision": float(prec),
            "direction_recall": float(rec),
            "direction_f1": float(f1),
            "arrow_seq_similarity": seq_sim,
            "direction_counts": {
                "total_ref_arrows": self.total_ref_arrows,
                "total_hyp_arrows": self.total_hyp_arrows,
                "total_tp": self.total_tp,
                "total_fp": int(self.total_hyp_arrows - self.total_tp),
                "total_fn": int(self.total_ref_arrows - self.total_tp),
            },
            "seq_eval": {
                "scope": self.seq_scope,
                "n_seq_eval": self.n_seq_eval,
            }
        }


# -------------------------
# Aggregators (CER/EM)
# -------------------------
class Agg:
    def __init__(self):
        self.n = 0
        self.sum_cer = 0.0
        self.sum_em = 0
        self.sum_dist = 0
        self.sum_ref_len = 0
        self.sum_characc = 0.0
        self.sum_characc_clipped = 0.0

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
    cuts = sorted(aggs_by_cut.keys())
    if not cuts:
        return {"n_cuts": 0}

    micro_cers = []
    ems = []
    for c in cuts:
        f = aggs_by_cut[c].finalize()
        micro_cers.append(float(f["cer_micro"]))
        ems.append(float(f["exact_match"]))

    return {
        "n_cuts": len(cuts),
        "macro_over_cuts__cer_micro_mean": mean(micro_cers),
        "macro_over_cuts__exact_match_mean": mean(ems),
    }

def macro_by_cut_arrows(aggs_by_cut: Dict[int, ArrowAgg]) -> Dict[str, Any]:
    cuts = sorted(aggs_by_cut.keys())
    if not cuts:
        return {"n_cuts": 0}

    pres = []
    cntx = []
    dprec = []
    drec = []
    df1 = []
    seq = []

    for c in cuts:
        f = aggs_by_cut[c].finalize()
        if f.get("arrow_presence_recall") is not None:
            pres.append(float(f["arrow_presence_recall"]))
        if f.get("arrow_count_exact") is not None:
            cntx.append(float(f["arrow_count_exact"]))
        dprec.append(float(f["direction_precision"]))
        drec.append(float(f["direction_recall"]))
        df1.append(float(f["direction_f1"]))
        if f.get("arrow_seq_similarity") is not None:
            seq.append(float(f["arrow_seq_similarity"]))

    return {
        "n_cuts": len(cuts),
        "macro_over_cuts__arrow_presence_recall_mean": safe_mean(pres),
        "macro_over_cuts__arrow_count_exact_mean": safe_mean(cntx),
        "macro_over_cuts__direction_precision_mean": safe_mean(dprec),
        "macro_over_cuts__direction_recall_mean": safe_mean(drec),
        "macro_over_cuts__direction_f1_mean": safe_mean(df1),
        "macro_over_cuts__arrow_seq_similarity_mean": safe_mean(seq),
    }


# -------------------------
# Stage1/Stage2 loaders
# -------------------------
S2_SPLIT_RE = re.compile(r"^cut(\d+)\.stage2\.split\.(nojp|jp)\.json$")
S1_RE = re.compile(r"^cut(\d+)\.stage1\.json$")

def build_stage2_split_path_map(stage2_split_dir: str) -> Dict[str, Dict[int, str]]:
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


# -------------------------
# Delta helpers
# -------------------------
def compute_delta_numeric(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        if k not in a or k not in b:
            continue
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            out[k] = None
            continue
        try:
            out[k] = float(vb) - float(va)
        except Exception:
            pass
    return out


# -------------------------
# Evaluation core (include failures + arrow metrics)
# -------------------------
def evaluate_subset(
    *,
    samples: List[Sample],
    stage1_path_map: Dict[int, str],
    stage2_paths_for_variant: Dict[int, str],
    neutral_symbols: List[str],
    arrow_set: List[str],
    arrow_seq_scope: str,
    subset_name: str,
    write_samples_jsonl: bool,
    out_jsonl_path: Optional[str],
) -> Dict[str, Any]:
    # caches
    s1_cut_cache: Dict[int, Dict[Tuple[int, int, str], str]] = {}
    s2_cut_cache: Dict[int, Dict[Tuple[int, int, str], Dict[str, str]]] = {}

    # text totals
    s1_total = Agg()
    s2_total = Agg()
    s1_by_col: Dict[str, Agg] = {}
    s2_by_col: Dict[str, Agg] = {}
    s1_by_cut: Dict[int, Agg] = {}
    s2_by_cut: Dict[int, Agg] = {}

    # arrow totals
    s1_arrow_total = ArrowAgg(arrow_set=arrow_set, seq_scope=arrow_seq_scope)
    s2_arrow_total = ArrowAgg(arrow_set=arrow_set, seq_scope=arrow_seq_scope)
    s1_arrow_by_col: Dict[str, ArrowAgg] = {}
    s2_arrow_by_col: Dict[str, ArrowAgg] = {}
    s1_arrow_by_cut: Dict[int, ArrowAgg] = {}
    s2_arrow_by_cut: Dict[int, ArrowAgg] = {}

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

            # ---- stage1 (must exist) ----
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
            if not isinstance(hyp_s1, str):
                hyp_s1 = str(hyp_s1)

            # ---- stage2 (optional; fallback to stage1) ----
            s2_path = stage2_paths_for_variant.get(cut)
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

            # ---- text metrics ----
            s1_total.add(hyp_s1, ref, neutral_symbols)
            s2_total.add(hyp_s2, ref, neutral_symbols)

            if sp.column not in s1_by_col:
                s1_by_col[sp.column] = Agg()
                s2_by_col[sp.column] = Agg()
            s1_by_col[sp.column].add(hyp_s1, ref, neutral_symbols)
            s2_by_col[sp.column].add(hyp_s2, ref, neutral_symbols)

            if cut not in s1_by_cut:
                s1_by_cut[cut] = Agg()
                s2_by_cut[cut] = Agg()
            s1_by_cut[cut].add(hyp_s1, ref, neutral_symbols)
            s2_by_cut[cut].add(hyp_s2, ref, neutral_symbols)

            # ---- arrow metrics ----
            s1_arrow_total.add(hyp_s1, ref)
            s2_arrow_total.add(hyp_s2, ref)

            if sp.column not in s1_arrow_by_col:
                s1_arrow_by_col[sp.column] = ArrowAgg(arrow_set=arrow_set, seq_scope=arrow_seq_scope)
                s2_arrow_by_col[sp.column] = ArrowAgg(arrow_set=arrow_set, seq_scope=arrow_seq_scope)
            s1_arrow_by_col[sp.column].add(hyp_s1, ref)
            s2_arrow_by_col[sp.column].add(hyp_s2, ref)

            if cut not in s1_arrow_by_cut:
                s1_arrow_by_cut[cut] = ArrowAgg(arrow_set=arrow_set, seq_scope=arrow_seq_scope)
                s2_arrow_by_cut[cut] = ArrowAgg(arrow_set=arrow_set, seq_scope=arrow_seq_scope)
            s1_arrow_by_cut[cut].add(hyp_s1, ref)
            s2_arrow_by_cut[cut].add(hyp_s2, ref)

            # ---- jsonl ----
            if jsonl_f is not None:
                if CFG["max_samples_jsonl"] is None or dumped < int(CFG["max_samples_jsonl"]):
                    cer1, _, _ = cer_and_dist(hyp_s1, ref, neutral_symbols)
                    cer2, _, _ = cer_and_dist(hyp_s2, ref, neutral_symbols)

                    ref_ar = extract_arrows(ref, arrow_set)
                    s1_ar = extract_arrows(hyp_s1, arrow_set)
                    s2_ar = extract_arrows(hyp_s2, arrow_set)

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
                        "arrows_ref": ref_ar,
                        "arrows_stage1": s1_ar,
                        "arrows_stage2": s2_ar,
                        "arrow_seq_sim_stage1": (
                            arrow_seq_similarity(s1_ar, ref_ar) if (len(ref_ar) > 0 or arrow_seq_scope == "ref_or_hyp_has_arrow") else None
                        ),
                        "arrow_seq_sim_stage2": (
                            arrow_seq_similarity(s2_ar, ref_ar) if (len(ref_ar) > 0 or arrow_seq_scope == "ref_or_hyp_has_arrow") else None
                        ),
                    }
                    jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    dumped += 1

        # finalize text
        s1_total_f = s1_total.finalize()
        s2_total_f = s2_total.finalize()
        s1_by_col_f = {c: a.finalize() for c, a in sorted(s1_by_col.items())}
        s2_by_col_f = {c: a.finalize() for c, a in sorted(s2_by_col.items())}
        s1_macro_cut = macro_by_cut(s1_by_cut)
        s2_macro_cut = macro_by_cut(s2_by_cut)

        # finalize arrows
        s1_arrow_total_f = s1_arrow_total.finalize()
        s2_arrow_total_f = s2_arrow_total.finalize()
        s1_arrow_by_col_f = {c: a.finalize() for c, a in sorted(s1_arrow_by_col.items())}
        s2_arrow_by_col_f = {c: a.finalize() for c, a in sorted(s2_arrow_by_col.items())}
        s1_arrow_macro_cut = macro_by_cut_arrows(s1_arrow_by_cut)
        s2_arrow_macro_cut = macro_by_cut_arrows(s2_arrow_by_cut)

        # deltas
        delta_text_total = compute_delta_numeric(
            s1_total_f, s2_total_f,
            keys=["cer_macro_over_samples", "cer_micro", "exact_match", "char_accuracy_macro", "char_accuracy_micro"]
        )

        delta_arrow_total = compute_delta_numeric(
            s1_arrow_total_f, s2_arrow_total_f,
            keys=[
                "arrow_presence_recall",
                "arrow_count_exact",
                "direction_precision",
                "direction_recall",
                "direction_f1",
                "arrow_seq_similarity",
            ]
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
                "delta_stage2_minus_stage1": delta_text_total,
            },
            "arrow_total": {
                "stage1_raw": s1_arrow_total_f,
                "stage2_corrected": s2_arrow_total_f,
                "delta_stage2_minus_stage1": delta_arrow_total,
            },
            "by_column": {
                "stage1_raw": s1_by_col_f,
                "stage2_corrected": s2_by_col_f,
            },
            "arrow_by_column": {
                "stage1_raw": s1_arrow_by_col_f,
                "stage2_corrected": s2_arrow_by_col_f,
            },
            "macro_by_cut": {
                "stage1_raw": s1_macro_cut,
                "stage2_corrected": s2_macro_cut,
            },
            "arrow_macro_by_cut": {
                "stage1_raw": s1_arrow_macro_cut,
                "stage2_corrected": s2_arrow_macro_cut,
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
        arrow_set=CFG["arrow_set"],
        arrow_seq_scope=str(CFG.get("arrow_seq_similarity_scope", "ref_has_arrow")),
        subset_name="train_test_cut_ids_only",
        write_samples_jsonl=CFG["write_samples_jsonl"],
        out_jsonl_path=out_jsonl_test,
    )

    res_all = evaluate_subset(
        samples=all_samples,
        stage1_path_map=stage1_path_map,
        stage2_paths_for_variant=stage2_paths_for_variant,
        neutral_symbols=CFG["neutral_symbols"],
        arrow_set=CFG["arrow_set"],
        arrow_seq_scope=str(CFG.get("arrow_seq_similarity_scope", "ref_has_arrow")),
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
            "arrow_set": CFG["arrow_set"],
            "arrow_seq_similarity_scope": CFG.get("arrow_seq_similarity_scope", "ref_has_arrow"),
            "note": "Stage2 missing cut/cell is evaluated by treating stage2 output == stage1 output (fallback). Arrow metrics added.",
        },
        "train_split": {
            "n_test_cut_ids": len(test_cut_ids),
        },
        "results": {
            "test_cut_ids_only": res_test,
            "all_cuts": res_all,
        }
    }

    out_path = os.path.join(out_dir, f"eval_stage2_vs_gt_split.include_failures.with_arrows.{variant}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    def _p(label: str, obj: Dict[str, Any]):
        t = obj["total"]
        a = obj["arrow_total"]
        s1 = t["stage1_raw"]
        s2 = t["stage2_corrected"]
        d = t["delta_stage2_minus_stage1"]
        c = obj["counts"]

        a1 = a["stage1_raw"]
        a2 = a["stage2_corrected"]
        ad = a["delta_stage2_minus_stage1"]

        print(f"\n=== {label} ({variant}) ===")
        print("samples_input:", c["samples_input"], " evaluated:", c["samples_evaluated"])
        print("missing_stage2_file:", c["missing_stage2_file"], " missing_stage2_cell:", c["missing_cell_in_stage2"],
              " fallback_used:", c["stage2_fallback_used_to_stage1"])

        print("[Text][Stage1 raw]  CER_micro=", s1["cer_micro"], " CER_macro=", s1["cer_macro_over_samples"], " EM=", s1["exact_match"])
        print("[Text][Stage2 corr] CER_micro=", s2["cer_micro"], " CER_macro=", s2["cer_macro_over_samples"], " EM=", s2["exact_match"])
        print("[Text][Delta S2-S1] ", d)

        print("[Arrow][Stage1 raw]  PresenceRec=", a1["arrow_presence_recall"],
              " CountExact=", a1["arrow_count_exact"],
              " DirF1=", a1["direction_f1"],
              " SeqSim=", a1["arrow_seq_similarity"])
        print("[Arrow][Stage2 corr] PresenceRec=", a2["arrow_presence_recall"],
              " CountExact=", a2["arrow_count_exact"],
              " DirF1=", a2["direction_f1"],
              " SeqSim=", a2["arrow_seq_similarity"])
        print("[Arrow][Delta S2-S1] ", ad)

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

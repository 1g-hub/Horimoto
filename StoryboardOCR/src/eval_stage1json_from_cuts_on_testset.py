# eval_stage1json_from_cuts_on_testset.py
# ------------------------------------------------------------
# 목적:
# - split_by_cut.json の test_cut_ids を使って "同じテスト分割" を再現
# - outputs/episode01/cuts/cut000n.stage1.json (stage1 OCR出力) を読み
#   annotation_table.csv の final_gt と突き合わせて CER/EM/矢印指標を算出
# - 矢印指標は
#     (A) picture/action_memo/dialogue の列別
#     (B) picture+action_memo+dialogue をまとめた合算
#   を両方出力
#
# 出力:
# - out_dir/
#   - metrics_stage1json_test.json
#   - (optional) records_stage1json_test.jsonl
# ------------------------------------------------------------

import os
import csv
import json
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # test split source (from training output)
    "split_by_cut_json": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train50per/split_by_cut.json",

    # stage1 outputs (cut0001.stage1.json etc)
    "cuts_dir": "outputs_withoutRAG/episode01/cuts",
    "cut_json_suffix": ".stage1.json",   # file suffix
    "cut_digits": 4,                     # cut0001

    # ground truth
    "annotation_csv": "data/episode01/annotation_table.csv",

    # evaluate these columns
    "eval_columns": ["cut", "picture", "action_memo", "dialogue", "time"],

    # arrow metrics columns
    "arrow_columns": ["picture", "action_memo", "dialogue"],

    # normalization for CER/EM
    "neutral_symbols": ["□", "△"],
    "space_insensitive": True,

    # optional cut range filter (match training setting)
    "select_cut_range": True,
    "cut_start": 1,
    "cut_end": 255,

    # output
    "out_dir": "outputs/episode01/stage1_ocr_lora_133arrow_focus_cut255_train50per/eval_stage1json_test",
    "dump_records_jsonl": True,  # per-cell records (can be big)
}
# =========================


ARROWS = ["→", "←", "↑", "↓"]
ARROW_RE = re.compile(r"[→←↑↓]")


# -------------------------
# small utils
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_int_str(x: Any) -> str:
    """Normalize numeric-like strings: '001' -> '1' """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        return str(int(s))
    except Exception:
        return s


def in_cut_range(cut_str: str) -> bool:
    if not CFG["select_cut_range"]:
        return True
    try:
        c = int(str(cut_str).strip())
    except Exception:
        return False
    return CFG["cut_start"] <= c <= CFG["cut_end"]


# -------------------------
# text metrics (CER/EM)
# -------------------------
def normalize_for_eval(s: str) -> str:
    s = s or ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for sym in CFG["neutral_symbols"]:
        s = s.replace(sym, "")
    if CFG["space_insensitive"]:
        s = " ".join(s.split())
    return s.strip()


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


def cer(hyp: str, ref: str) -> float:
    h = list(normalize_for_eval(hyp))
    r = list(normalize_for_eval(ref))
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return levenshtein(h, r) / len(r)


def exact_match(hyp: str, ref: str) -> int:
    return int(normalize_for_eval(hyp) == normalize_for_eval(ref))


# -------------------------
# arrow metrics (ref_has only)
# -------------------------
def extract_arrow_seq(text: str) -> List[str]:
    return ARROW_RE.findall(text or "")


def arrow_metrics_from_pairs_refhas(pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    pairs: list of (ref, hyp)
    - "ref に矢印があるサンプルのみ" で集計
    - direction は順序無視(マルチセット)で micro 集計
    - seq similarity は 1 - ed/max(|ref|,|hyp|,1) で 0..1
    """
    filtered: List[Tuple[List[str], List[str]]] = []
    for ref, hyp in pairs:
        rseq = extract_arrow_seq(ref)
        if len(rseq) == 0:
            continue
        hseq = extract_arrow_seq(hyp)
        filtered.append((rseq, hseq))

    n = len(filtered)
    out: Dict[str, Any] = {
        "n_samples": float(n),
        "n_ref_has_arrow": float(n),
    }
    if n == 0:
        out.update({
            "arrow_presence_recall_if_ref_has": 0.0,
            "arrow_count_exact_if_ref_has": 0.0,
            "arrow_count_mae_if_ref_has": 0.0,
            "arrow_direction_precision_micro": 0.0,
            "arrow_direction_recall_micro": 0.0,
            "arrow_direction_f1_micro": 0.0,
            "arrow_seq_exact_if_ref_has": 0.0,
            "arrow_seq_edit_rate_micro": 0.0,
            "arrow_seq_similarity_mean_0_1": 0.0,
        })
        return out

    presence_hits = 0
    count_exact = 0
    count_abs_err_sum = 0.0
    seq_exact = 0

    tp = 0
    fp = 0
    fn = 0

    total_edit = 0
    total_ref_len = 0
    sim_sum = 0.0

    for rseq, hseq in filtered:
        if len(hseq) > 0:
            presence_hits += 1

        if len(hseq) == len(rseq):
            count_exact += 1
        count_abs_err_sum += abs(len(hseq) - len(rseq))

        if hseq == rseq:
            seq_exact += 1

        rc = Counter(rseq)
        hc = Counter(hseq)
        for a in ARROWS:
            tp += min(rc.get(a, 0), hc.get(a, 0))
            fp += max(hc.get(a, 0) - rc.get(a, 0), 0)
            fn += max(rc.get(a, 0) - hc.get(a, 0), 0)

        ed = levenshtein(hseq, rseq)
        total_edit += ed
        total_ref_len += len(rseq)

        denom = max(len(rseq), len(hseq), 1)
        sim_sum += 1.0 - (ed / denom)

    out["arrow_presence_recall_if_ref_has"] = presence_hits / n
    out["arrow_count_exact_if_ref_has"] = count_exact / n
    out["arrow_count_mae_if_ref_has"] = count_abs_err_sum / n

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    out["arrow_direction_precision_micro"] = prec
    out["arrow_direction_recall_micro"] = rec
    out["arrow_direction_f1_micro"] = f1

    out["arrow_seq_exact_if_ref_has"] = seq_exact / n
    out["arrow_seq_edit_rate_micro"] = (total_edit / total_ref_len) if total_ref_len > 0 else 0.0
    out["arrow_seq_similarity_mean_0_1"] = sim_sum / n
    return out


# -------------------------
# IO
# -------------------------
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# load split + gt map
# -------------------------
def load_test_cut_ids(split_path: str) -> List[str]:
    split = read_json(split_path)
    test_cut_ids = (split.get("split_info", {}) or {}).get("test_cut_ids", [])
    if not test_cut_ids:
        test_cut_ids = split.get("test_cut_ids", [])
    # normalize
    out = []
    for c in test_cut_ids:
        cs = to_int_str(c)
        if cs and in_cut_range(cs):
            out.append(cs)
    # de-dup preserving order
    seen = set()
    out2 = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        out2.append(x)
    return out2


def build_gt_map(annotation_csv: str) -> Dict[Tuple[str, str, str, str], str]:
    """
    key: (cut_str, page_str, row_str, column)
    value: final_gt
    """
    rows = read_csv_rows(annotation_csv)
    gt_map: Dict[Tuple[str, str, str, str], str] = {}
    for r in rows:
        col = (r.get("column") or "").strip()
        if col not in set(CFG["eval_columns"]):
            continue
        cut = to_int_str(r.get("cut"))
        page = to_int_str(r.get("page"))
        row = to_int_str(r.get("row"))
        if not cut or not page or not row:
            continue
        if not in_cut_range(cut):
            continue
        gt = (r.get("final_gt") or "")
        if not gt.strip():
            continue
        k = (cut, page, row, col)
        # if duplicates exist, keep the latest (usually identical)
        gt_map[k] = gt
    return gt_map


def cut_to_stage1_path(cut_str: str) -> str:
    # prefer numeric padding
    try:
        c = int(cut_str)
        name = f"cut{c:0{CFG['cut_digits']}d}{CFG['cut_json_suffix']}"
        p = os.path.join(CFG["cuts_dir"], name)
        if os.path.isfile(p):
            return p
        # fallback: without padding
        p2 = os.path.join(CFG["cuts_dir"], f"cut{cut_str}{CFG['cut_json_suffix']}")
        return p2
    except Exception:
        return os.path.join(CFG["cuts_dir"], f"cut{cut_str}{CFG['cut_json_suffix']}")


# -------------------------
# main eval
# -------------------------
def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)

    test_cut_ids = load_test_cut_ids(CFG["split_by_cut_json"])
    if not test_cut_ids:
        raise RuntimeError("test_cut_ids is empty. Check split_by_cut.json.")

    gt_map = build_gt_map(CFG["annotation_csv"])
    if not gt_map:
        raise RuntimeError("GT map is empty. Check annotation_table.csv path and filters.")

    # collect per-cell records
    records: List[Dict[str, Any]] = []
    missing_cut_files = 0
    missing_gt = 0

    for cut_str in test_cut_ids:
        cut_path = cut_to_stage1_path(cut_str)
        if not os.path.isfile(cut_path):
            missing_cut_files += 1
            continue

        try:
            cut_obj = read_json(cut_path)
        except Exception as e:
            print(f"[WARN] failed to read: {cut_path} ({e})")
            continue

        # normalize cut from json if present
        cut_norm = to_int_str(cut_obj.get("cut_str") or cut_obj.get("cut") or cut_str) or cut_str

        rows = cut_obj.get("rows") or []
        for row_obj in rows:
            page = to_int_str(row_obj.get("page"))
            rowi = to_int_str(row_obj.get("row"))
            cols = row_obj.get("cols") or {}

            if not page or not rowi:
                continue

            for col in CFG["eval_columns"]:
                if col not in cols:
                    continue
                cell = cols.get(col) or {}
                hyp = cell.get("raw_text") or ""
                variant = cell.get("model_variant") or ""

                key = (cut_norm, page, rowi, col)
                ref = gt_map.get(key)

                if ref is None:
                    missing_gt += 1
                    continue

                c = cer(hyp, ref)
                em = exact_match(hyp, ref)

                records.append({
                    "cut": cut_norm,
                    "page": page,
                    "row": rowi,
                    "column": col,
                    "model_variant": variant,
                    "ref": ref,
                    "hyp": hyp,
                    "cer": c,
                    "exact_match": em,
                    "ref_arrow_seq": extract_arrow_seq(ref),
                    "hyp_arrow_seq": extract_arrow_seq(hyp),
                    "stage1_json": cut_path,
                    "cell_image": (cell.get("image") or ""),
                })

    if not records:
        raise RuntimeError("No records matched between stage1 outputs and annotation_table.csv.")

    # text metrics agg
    total_n = len(records)
    cer_mean = sum(r["cer"] for r in records) / max(total_n, 1)
    em_mean = sum(r["exact_match"] for r in records) / max(total_n, 1)

    by_col = defaultdict(lambda: {"n": 0, "cer_sum": 0.0, "em_sum": 0})
    for r in records:
        a = by_col[r["column"]]
        a["n"] += 1
        a["cer_sum"] += r["cer"]
        a["em_sum"] += r["exact_match"]

    text_by_column = {}
    for col, a in by_col.items():
        n = max(a["n"], 1)
        text_by_column[col] = {
            "n": a["n"],
            "cer": a["cer_sum"] / n,
            "exact_match": a["em_sum"] / n,
        }

    # arrow metrics: by column (ref_has only)
    arrow_by_column_refhas = {}
    for col in CFG["arrow_columns"]:
        pairs = [(r["ref"], r["hyp"]) for r in records if r["column"] == col]
        arrow_by_column_refhas[col] = arrow_metrics_from_pairs_refhas(pairs)

    # arrow metrics: combined picture+action_memo+dialogue (ref_has only)
    arrow_cols_set = set(CFG["arrow_columns"])
    pairs_combined = [(r["ref"], r["hyp"]) for r in records if r["column"] in arrow_cols_set]
    arrow_combined_refhas = arrow_metrics_from_pairs_refhas(pairs_combined)

    # also: overall ref_has (all eval columns)
    pairs_all = [(r["ref"], r["hyp"]) for r in records]
    arrow_allcols_refhas = arrow_metrics_from_pairs_refhas(pairs_all)

    out = {
        "meta": {
            "created_at": now_iso(),
            "split_by_cut_json": CFG["split_by_cut_json"],
            "cuts_dir": CFG["cuts_dir"],
            "annotation_csv": CFG["annotation_csv"],
            "eval_columns": CFG["eval_columns"],
            "arrow_columns": CFG["arrow_columns"],
            "neutral_symbols": CFG["neutral_symbols"],
            "space_insensitive": CFG["space_insensitive"],
            "cut_range_filter": {
                "select_cut_range": CFG["select_cut_range"],
                "cut_start": CFG["cut_start"],
                "cut_end": CFG["cut_end"],
            },
            "test_cut_ids_count": len(test_cut_ids),
            "stage1_cut_files_missing": missing_cut_files,
            "cells_missing_gt": missing_gt,
            "cells_evaluated": total_n,
        },
        "text_metrics": {
            "total_macro": {
                "n": total_n,
                "cer": cer_mean,
                "exact_match": em_mean,
            },
            "by_column_macro": text_by_column,
        },
        "arrow_metrics_ref_has_only": {
            "by_column": arrow_by_column_refhas,
            "combined_picture_action_dialogue": arrow_combined_refhas,
            "all_eval_columns": arrow_allcols_refhas,
        },
    }

    out_path = os.path.join(CFG["out_dir"], "metrics_stage1json_test.json")
    write_json(out_path, out)
    print(f"[OK] wrote {out_path}")

    if CFG["dump_records_jsonl"]:
        jsonl_path = os.path.join(CFG["out_dir"], "records_stage1json_test.jsonl")
        write_jsonl(jsonl_path, records)
        print(f"[OK] wrote {jsonl_path}")

    # short console summary
    print("\n=== TEXT (macro) ===")
    print("CER:", out["text_metrics"]["total_macro"]["cer"])
    print("EM :", out["text_metrics"]["total_macro"]["exact_match"])

    print("\n=== ARROW (ref_has only) combined picture/action_memo/dialogue ===")
    for k, v in out["arrow_metrics_ref_has_only"]["combined_picture_action_dialogue"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n=== ARROW (ref_has only) by column ===")
    for col in CFG["arrow_columns"]:
        m = out["arrow_metrics_ref_has_only"]["by_column"][col]
        print(f"\n[{col}] n_ref_has={int(m.get('n_ref_has_arrow', 0))}")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()

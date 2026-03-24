# eval_compare_base_vs_lora_test_plus_arrows_bycol.py
# ------------------------------------------------------------
# Compare BASE vs LoRA on the SAME test set (split_by_cut.json),
# compute CER / ExactMatch (space-insensitive, neutral symbols removed),
# PLUS arrow-specific metrics:
#  - Column-wise arrow metrics (picture/action_memo/dialogue)
#  - Bar charts ONLY on samples where reference contains arrows
#  - Top-N arrow error examples: miss / direction mismatch / over-detect
#    + "LoRA-new errors" (base ok but lora fails)
# ------------------------------------------------------------

import os
import csv
import json
import shutil
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # Base model (must match training base)
    "base_model_id": "Qwen/Qwen3-VL-4B-Instruct",

    # LoRA adapter directory (output_dir of training)
    "lora_adapter_dir": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train50per/adapter",

    # CSV with final_gt (and cut/page/row/column)
    "csv_path": "data/episode01/annotation_table.csv",
    "episode_dir": "data/episode01",

    # split_by_cut.json from training output_dir (to ensure SAME test set)
    "split_by_cut_json": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train50per/split_by_cut.json",

    # Optional: restrict evaluation to cut range (useful if training used cut range)
    "select_cut_range": True,
    "cut_start": 1,
    "cut_end": 255,

    # Neutral symbols removed before CER/ExactMatch (space-insensitive)
    "neutral_symbols": ["□", "△"],

    # Evaluate only these columns for text metrics (CER/ExactMatch)
    # None => all
    "eval_columns": {"cut", "time", "action_memo", "dialogue", "picture"},

    # Arrow metrics are reported by these columns
    "arrow_columns_order": ["picture", "action_memo", "dialogue"],

    # Generation settings (deterministic)
    "gen": {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    },

    # Prompt
    "use_column_conditioned_prompt": True,

    # Limit number of test samples (None => all)
    "max_samples": None,

    # Output base dir
    "out_dir": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train50per/eval_compare_arrows",

    # Top-N examples to export per category
    "top_n": 30,

    # If True, copy images into output folders for quick review (optional)
    "copy_images": False,

    # If True, also dump per-sample JSONL with arrow details (can be big)
    "dump_all_samples_jsonl": True,
}
# =========================


# -------------------------
# Normalization + metrics (space-insensitive + neutral removal)
# -------------------------
def normalize_for_eval(s: str, neutral_symbols: List[str]) -> str:
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


def cer(hyp: str, ref: str, neutral_symbols: List[str]) -> float:
    hyp2 = normalize_for_eval(hyp, neutral_symbols)
    ref2 = normalize_for_eval(ref, neutral_symbols)
    h = list(hyp2)
    r = list(ref2)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return levenshtein(h, r) / len(r)


def exact_match(hyp: str, ref: str, neutral_symbols: List[str]) -> int:
    return int(normalize_for_eval(hyp, neutral_symbols) == normalize_for_eval(ref, neutral_symbols))


# -------------------------
# Arrow-only evaluation
# -------------------------
ARROW_CHARS = ["→", "←", "↑", "↓"]
ARROW_RE = re.compile(r"[→←↑↓]")


def extract_arrow_seq(text: str) -> List[str]:
    return ARROW_RE.findall(text or "")


def arrow_counter(seq: List[str]) -> Counter:
    return Counter(seq)


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def arrow_metrics_one(hyp: str, ref: str) -> Dict[str, Any]:
    """
    1サンプル分の矢印評価（→←↑↓）
    - refに矢印があるかどうかで条件付き指標を返す
    - 方向一致は「順序なし」のマルチセット一致でTP/FP/FNを計算
    - シーケンス一致は順序も見る
    """
    ref_seq = extract_arrow_seq(ref)
    hyp_seq = extract_arrow_seq(hyp)

    ref_cnt = len(ref_seq)
    hyp_cnt = len(hyp_seq)

    ref_has = int(ref_cnt > 0)
    hyp_has = int(hyp_cnt > 0)

    rc = arrow_counter(ref_seq)
    hc = arrow_counter(hyp_seq)

    tp = 0
    fp = 0
    fn = 0
    per_dir = {}

    for d in ARROW_CHARS:
        r = int(rc.get(d, 0))
        h = int(hc.get(d, 0))
        tpd = min(r, h)
        fpd = max(h - r, 0)
        fnd = max(r - h, 0)
        tp += tpd
        fp += fpd
        fn += fnd
        per_dir[d] = {
            "ref": r, "hyp": h, "tp": tpd, "fp": fpd, "fn": fnd,
            "recall_dir": safe_div(tpd, r),
            "precision_dir": safe_div(tpd, h),
        }

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # sequence (order-sensitive)
    seq_exact = int(ref_seq == hyp_seq)
    ed = levenshtein(hyp_seq, ref_seq)

    # edit rate wrt ref (can exceed 1 if hyp is much longer)
    seq_edit_rate = safe_div(ed, ref_cnt) if ref_cnt > 0 else (0.0 if hyp_cnt == 0 else 1.0)

    # 0..1 similarity (always within range)
    denom = max(ref_cnt, hyp_cnt, 1)
    seq_similarity = 1.0 - (ed / denom)

    # conditionals
    presence_hit_if_ref_has = int(ref_has == 1 and hyp_has == 1)
    count_exact_if_ref_has = int(ref_has == 1 and hyp_cnt == ref_cnt)
    count_abs_error_if_ref_has = abs(hyp_cnt - ref_cnt) if ref_has == 1 else 0

    seq_exact_if_ref_has = int(ref_has == 1 and seq_exact == 1)

    return {
        "ref_arrow_seq": ref_seq,
        "hyp_arrow_seq": hyp_seq,

        "ref_has_arrow": ref_has,
        "hyp_has_arrow": hyp_has,

        "ref_arrow_count": ref_cnt,
        "hyp_arrow_count": hyp_cnt,

        "presence_hit_if_ref_has": presence_hit_if_ref_has,
        "count_exact_if_ref_has": count_exact_if_ref_has,
        "count_abs_error_if_ref_has": count_abs_error_if_ref_has,

        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_direction": per_dir,

        "seq_exact_if_ref_has": seq_exact_if_ref_has,
        "seq_edit_distance": ed,
        "seq_edit_rate": seq_edit_rate,
        "seq_similarity_0_1": seq_similarity,
    }


class ArrowMetricsAggregator:
    """
    Micro集計（矢印専用）
    - update() されたサンプル集合に対して指標を出す
    """
    def __init__(self):
        self.n = 0
        self.n_ref_has = 0

        self.presence_hits = 0
        self.count_exact = 0
        self.count_abs_err_sum = 0

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.seq_exact = 0
        self.seq_ed_sum = 0
        self.seq_ref_len_sum = 0
        self.seq_sim_sum = 0.0

    def update(self, hyp: str, ref: str):
        m = arrow_metrics_one(hyp, ref)
        self.n += 1

        if m["ref_has_arrow"] == 1:
            self.n_ref_has += 1
            self.presence_hits += m["presence_hit_if_ref_has"]
            self.count_exact += m["count_exact_if_ref_has"]
            self.count_abs_err_sum += m["count_abs_error_if_ref_has"]

            self.seq_exact += m["seq_exact_if_ref_has"]
            self.seq_ed_sum += m["seq_edit_distance"]
            self.seq_ref_len_sum += m["ref_arrow_count"]
            self.seq_sim_sum += m["seq_similarity_0_1"]

        self.tp += m["tp"]
        self.fp += m["fp"]
        self.fn += m["fn"]

    def finalize(self) -> Dict[str, Any]:
        presence_recall = safe_div(self.presence_hits, self.n_ref_has)
        count_exact = safe_div(self.count_exact, self.n_ref_has)
        count_mae = safe_div(self.count_abs_err_sum, self.n_ref_has)

        seq_exact = safe_div(self.seq_exact, self.n_ref_has)
        seq_edit_rate_micro = safe_div(self.seq_ed_sum, self.seq_ref_len_sum)
        seq_similarity_mean = safe_div(self.seq_sim_sum, self.n_ref_has)

        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        return {
            "n_samples": self.n,
            "n_ref_has_arrow": self.n_ref_has,

            "arrow_presence_recall_if_ref_has": presence_recall,
            "arrow_count_exact_if_ref_has": count_exact,
            "arrow_count_mae_if_ref_has": count_mae,

            "arrow_direction_precision_micro": precision,
            "arrow_direction_recall_micro": recall,
            "arrow_direction_f1_micro": f1,

            "arrow_seq_exact_if_ref_has": seq_exact,
            "arrow_seq_edit_rate_micro": seq_edit_rate_micro,
            "arrow_seq_similarity_mean_0_1": seq_similarity_mean,
        }


def classify_arrow_error(ref_seq: List[str], hyp_seq: List[str], tp: int) -> Optional[str]:
    """
    エラーカテゴリ（refに矢印があるケースを主に分類）
    - miss: ref>0 and hyp==0
    - over_detect: hyp>ref
    - direction_mismatch: ref>0, hyp>0, hyp<=ref, and tp < hyp  (予測した矢印の中に方向不一致がある)
    - under_detect: 0<hyp<ref (参考)
    - None: それ以外（完全一致に近い）
    """
    ref_cnt = len(ref_seq)
    hyp_cnt = len(hyp_seq)

    if ref_cnt == 0:
        return None

    if hyp_cnt == 0:
        return "miss"

    if hyp_cnt > ref_cnt:
        return "over_detect"

    # hyp_cnt <= ref_cnt and hyp_cnt>0
    # tp < hyp_cnt なら、予測した矢印の中に方向が合わないものが含まれる
    if tp < hyp_cnt:
        return "direction_mismatch"

    # ここまで来て hyp_cnt < ref_cnt なら不足だが方向は合っている
    if hyp_cnt < ref_cnt:
        return "under_detect"

    return None


# -------------------------
# Prompt
# -------------------------
def build_single_prompt(column: str) -> str:
    return (
        "You are an OCR engine for anime storyboards.\n"
        f"Target column: {column}.\n"
        "The storyboard contains BOTH handwritten and computer-typed text.\n"
        "Handwritten text often includes arrows and short production terms.\n\n"
        "Task: Transcribe ALL visible text as faithfully as possible.\n"
        "Rules:\n"
        "- Keep original line breaks.\n"
        "- Preserve symbols exactly (→ ← ↑ ↓, brackets, punctuation).\n"
        "- Do NOT correct typos.\n"
        "- Do NOT translate.\n"
        "- If unreadable, output '□'.\n"
        "- Return ONLY the transcribed text. No explanations.\n"
    )


def reconstruct_image_path(episode_dir: str, column: str, page: str, row: str) -> str:
    return os.path.join(episode_dir, column, f"page{page}_row{row}.png")


# -------------------------
# Data selection
# -------------------------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_safe(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan"}:
        return None
    try:
        return int(s)
    except Exception:
        return None


def in_cut_range(cut_str: str) -> bool:
    if not CFG["select_cut_range"]:
        return True
    c = parse_int_safe(cut_str)
    if c is None:
        return False
    return CFG["cut_start"] <= c <= CFG["cut_end"]


def filter_test_rows(rows: List[Dict[str, str]], test_cut_ids: List[str]) -> List[Dict[str, str]]:
    test_set = set(test_cut_ids)
    out = []
    for r in rows:
        cut = r.get("cut", "")
        if cut not in test_set:
            continue
        if not in_cut_range(cut):
            continue
        if CFG["eval_columns"] and r.get("column", "") not in CFG["eval_columns"]:
            continue
        gt = (r.get("final_gt") or "").strip()
        if not gt:
            continue
        if not r.get("page") or not r.get("row"):
            continue
        # image existence checked later
        out.append(r)
    return out


# -------------------------
# Model loading
# -------------------------
def load_base_model() -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(CFG["base_model_id"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"],
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    return model, processor


def load_lora_model() -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    if PeftModel is None:
        raise RuntimeError("peft is not installed but required for LoRA evaluation.")
    processor = AutoProcessor.from_pretrained(CFG["base_model_id"])
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"],
        device_map="auto",
        torch_dtype="auto",
    )
    if not os.path.isdir(CFG["lora_adapter_dir"]):
        raise FileNotFoundError(f"LoRA adapter dir not found: {CFG['lora_adapter_dir']}")
    model = PeftModel.from_pretrained(base, CFG["lora_adapter_dir"])
    model.eval()
    return model, processor


@torch.no_grad()
def run_one(model, processor, image: Image.Image, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    out_ids = model.generate(
        **batch,
        pad_token_id=pad_id,
        **CFG["gen"],
    )
    # prompt part trimming
    prompt_len = batch["input_ids"].shape[1]
    trimmed = out_ids[0][prompt_len:]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()
    return text


# -------------------------
# Aggregation helpers (text metrics)
# -------------------------
def agg_init() -> Dict[str, Any]:
    return {"n": 0, "cer_sum": 0.0, "em_sum": 0}


def update_agg(agg: Dict[str, Any], hyp: str, ref: str, neutral: List[str]):
    agg["n"] += 1
    agg["cer_sum"] += cer(hyp, ref, neutral)
    agg["em_sum"] += exact_match(hyp, ref, neutral)


def finalize_agg(agg: Dict[str, Any]) -> Dict[str, float]:
    n = max(agg["n"], 1)
    return {"n": agg["n"], "cer": agg["cer_sum"] / n, "exact_match": agg["em_sum"] / n}


# -------------------------
# Plotting (bar charts)
# -------------------------
def plot_bar_base_vs_lora(
    *,
    columns_order: List[str],
    base_vals: List[float],
    lora_vals: List[float],
    ylabel: str,
    title: str,
    out_path: str,
):
    x = list(range(len(columns_order)))
    width = 0.35

    plt.figure(figsize=(10, 4))
    b1 = plt.bar([i - width / 2 for i in x], base_vals, width, label="base")
    b2 = plt.bar([i + width / 2 for i in x], lora_vals, width, label="lora")

    plt.xticks(x, columns_order)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    ax = plt.gca()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.savefig(out_path)
    plt.close()


# -------------------------
# IO helpers
# -------------------------
def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def maybe_copy_image(img_path: str, dst_dir: str, name: str):
    if not CFG["copy_images"]:
        return
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.isfile(img_path):
        shutil.copy2(img_path, os.path.join(dst_dir, name))


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)

    split = read_json(CFG["split_by_cut_json"])

    # split_info 内に入っている形式に対応
    test_cut_ids = (split.get("split_info", {}) or {}).get("test_cut_ids", [])
    if not test_cut_ids:
        # 念のため直下も見る（他の形式への互換）
        test_cut_ids = split.get("test_cut_ids", [])

    if not test_cut_ids:
        raise RuntimeError(
            "test_cut_ids is empty in split_by_cut.json.\n"
            f"keys={list(split.keys())}"
        )


    rows_all = read_csv_rows(CFG["csv_path"])
    test_rows = filter_test_rows(rows_all, test_cut_ids)
    if CFG["max_samples"] is not None:
        test_rows = test_rows[: int(CFG["max_samples"])]

    if not test_rows:
        raise RuntimeError("No test rows selected. Check CSV, split_by_cut.json, and cut range filter.")

    # Load models
    base_model, base_proc = load_base_model()
    lora_model, lora_proc = load_lora_model()

    # Text metrics aggregation
    base_total = agg_init()
    lora_total = agg_init()
    base_by_col: Dict[str, Any] = {}
    lora_by_col: Dict[str, Any] = {}

    # Arrow metrics aggregation:
    # (A) overall across evaluated samples
    arrow_all_base = ArrowMetricsAggregator()
    arrow_all_lora = ArrowMetricsAggregator()

    # (B) only samples where ref has arrows (for "矢印が含まれるサンプルだけ" summary)
    arrow_refhas_base = ArrowMetricsAggregator()
    arrow_refhas_lora = ArrowMetricsAggregator()

    # (C) column-wise arrow metrics (ref_has subset) for picture/action_memo/dialogue
    arrow_cols = CFG["arrow_columns_order"]
    arrow_bycol_refhas_base: Dict[str, ArrowMetricsAggregator] = {c: ArrowMetricsAggregator() for c in arrow_cols}
    arrow_bycol_refhas_lora: Dict[str, ArrowMetricsAggregator] = {c: ArrowMetricsAggregator() for c in arrow_cols}

    # Collect per-sample results
    sample_results: List[Dict[str, Any]] = []

    for idx, r in enumerate(tqdm(test_rows, desc="Eval test: base vs lora (+arrow metrics)")):
        col = r["column"]
        page = r["page"]
        row = r["row"]
        cut = r["cut"]
        ref = r["final_gt"]

        img_path = reconstruct_image_path(CFG["episode_dir"], col, page, row)
        if not os.path.isfile(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        prompt = build_single_prompt(col) if CFG["use_column_conditioned_prompt"] else build_single_prompt("unknown")

        hyp_base = run_one(base_model, base_proc, image, prompt)
        hyp_lora = run_one(lora_model, lora_proc, image, prompt)

        # ---- text metrics
        update_agg(base_total, hyp_base, ref, CFG["neutral_symbols"])
        update_agg(lora_total, hyp_lora, ref, CFG["neutral_symbols"])

        if col not in base_by_col:
            base_by_col[col] = agg_init()
            lora_by_col[col] = agg_init()
        update_agg(base_by_col[col], hyp_base, ref, CFG["neutral_symbols"])
        update_agg(lora_by_col[col], hyp_lora, ref, CFG["neutral_symbols"])

        cer_b = cer(hyp_base, ref, CFG["neutral_symbols"])
        cer_l = cer(hyp_lora, ref, CFG["neutral_symbols"])
        em_b = exact_match(hyp_base, ref, CFG["neutral_symbols"])
        em_l = exact_match(hyp_lora, ref, CFG["neutral_symbols"])

        # ---- arrow metrics (per sample)
        am_base = arrow_metrics_one(hyp_base, ref)
        am_lora = arrow_metrics_one(hyp_lora, ref)

        arrow_all_base.update(hyp_base, ref)
        arrow_all_lora.update(hyp_lora, ref)

        if am_base["ref_has_arrow"] == 1:
            arrow_refhas_base.update(hyp_base, ref)
            arrow_refhas_lora.update(hyp_lora, ref)

            if col in arrow_cols:
                arrow_bycol_refhas_base[col].update(hyp_base, ref)
                arrow_bycol_refhas_lora[col].update(hyp_lora, ref)

        # classify arrow errors (for Top-N)
        err_base = classify_arrow_error(am_base["ref_arrow_seq"], am_base["hyp_arrow_seq"], am_base["tp"])
        err_lora = classify_arrow_error(am_lora["ref_arrow_seq"], am_lora["hyp_arrow_seq"], am_lora["tp"])

        # "base is OK" definition for arrow correctness (strict-ish):
        # - if ref has arrows: base must have same count AND no direction mismatch (tp==hyp_cnt) AND seq exact
        #   (seq exact is strict; if you want loosen, remove it)
        ref_seq = am_base["ref_arrow_seq"]
        base_ok = False
        lora_ok = False
        if len(ref_seq) > 0:
            base_ok = (
                len(am_base["hyp_arrow_seq"]) == len(ref_seq)
                and am_base["tp"] == len(ref_seq)
                and am_base["hyp_arrow_seq"] == ref_seq
            )
            lora_ok = (
                len(am_lora["hyp_arrow_seq"]) == len(ref_seq)
                and am_lora["tp"] == len(ref_seq)
                and am_lora["hyp_arrow_seq"] == ref_seq
            )

        sample_results.append({
            "id": idx,
            "cut": cut,
            "page": page,
            "row": row,
            "column": col,
            "image_path": img_path,

            "ref": ref,
            "hyp_base": hyp_base,
            "hyp_lora": hyp_lora,

            "cer_base": cer_b,
            "cer_lora": cer_l,
            "delta_cer_lora_minus_base": cer_l - cer_b,
            "exact_base": em_b,
            "exact_lora": em_l,

            # arrows (store both raw seq and joined string for readability)
            "ref_arrow_seq": am_base["ref_arrow_seq"],
            "base_arrow_seq": am_base["hyp_arrow_seq"],
            "lora_arrow_seq": am_lora["hyp_arrow_seq"],
            "ref_arrow_str": "".join(am_base["ref_arrow_seq"]),
            "base_arrow_str": "".join(am_base["hyp_arrow_seq"]),
            "lora_arrow_str": "".join(am_lora["hyp_arrow_seq"]),

            "ref_arrow_count": am_base["ref_arrow_count"],
            "base_arrow_count": am_base["hyp_arrow_count"],
            "lora_arrow_count": am_lora["hyp_arrow_count"],

            "arrow_tp_base": am_base["tp"],
            "arrow_fp_base": am_base["fp"],
            "arrow_fn_base": am_base["fn"],
            "arrow_f1_base": am_base["f1"],
            "arrow_seq_similarity_base": am_base["seq_similarity_0_1"],
            "arrow_seq_exact_base_if_ref_has": am_base["seq_exact_if_ref_has"],

            "arrow_tp_lora": am_lora["tp"],
            "arrow_fp_lora": am_lora["fp"],
            "arrow_fn_lora": am_lora["fn"],
            "arrow_f1_lora": am_lora["f1"],
            "arrow_seq_similarity_lora": am_lora["seq_similarity_0_1"],
            "arrow_seq_exact_lora_if_ref_has": am_lora["seq_exact_if_ref_has"],

            "arrow_error_base": err_base,
            "arrow_error_lora": err_lora,
            "arrow_base_ok_strict": int(base_ok),
            "arrow_lora_ok_strict": int(lora_ok),

            # optional: per-direction counts (can be helpful)
            "arrow_per_direction_base": am_base["per_direction"],
            "arrow_per_direction_lora": am_lora["per_direction"],
        })

    # Finalize text metrics
    base_total_f = finalize_agg(base_total)
    lora_total_f = finalize_agg(lora_total)
    base_cols_f = {k: finalize_agg(v) for k, v in base_by_col.items()}
    lora_cols_f = {k: finalize_agg(v) for k, v in lora_by_col.items()}

    # Finalize arrow metrics
    arrow_all_base_f = arrow_all_base.finalize()
    arrow_all_lora_f = arrow_all_lora.finalize()
    arrow_refhas_base_f = arrow_refhas_base.finalize()
    arrow_refhas_lora_f = arrow_refhas_lora.finalize()

    arrow_bycol_refhas_base_f = {c: arrow_bycol_refhas_base[c].finalize() for c in arrow_cols}
    arrow_bycol_refhas_lora_f = {c: arrow_bycol_refhas_lora[c].finalize() for c in arrow_cols}

    # Save main comparison json
    out_compare = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model_id": CFG["base_model_id"],
            "lora_adapter_dir": CFG["lora_adapter_dir"],
            "csv_path": CFG["csv_path"],
            "split_by_cut_json": CFG["split_by_cut_json"],
            "test_cut_ids_count": len(test_cut_ids),
            "test_samples_used": int(base_total_f["n"]),
            "neutral_symbols": CFG["neutral_symbols"],
            "gen": CFG["gen"],
            "cut_range_filter": {
                "select_cut_range": CFG["select_cut_range"],
                "cut_start": CFG["cut_start"],
                "cut_end": CFG["cut_end"],
            }
        },
        "text_metrics": {
            "total_micro": {
                "base": base_total_f,
                "lora": lora_total_f,
                "delta_lora_minus_base": {
                    "cer": lora_total_f["cer"] - base_total_f["cer"],
                    "exact_match": lora_total_f["exact_match"] - base_total_f["exact_match"],
                }
            },
            "by_column": {
                "base": base_cols_f,
                "lora": lora_cols_f,
            }
        },
        "arrow_metrics": {
            "all_samples": {
                "base": arrow_all_base_f,
                "lora": arrow_all_lora_f,
                "delta_lora_minus_base": {
                    k: (arrow_all_lora_f[k] - arrow_all_base_f[k])
                    for k in arrow_all_base_f.keys()
                    if isinstance(arrow_all_base_f[k], float)
                }
            },
            "ref_has_arrows_only": {
                "base": arrow_refhas_base_f,
                "lora": arrow_refhas_lora_f,
                "delta_lora_minus_base": {
                    k: (arrow_refhas_lora_f[k] - arrow_refhas_base_f[k])
                    for k in arrow_refhas_base_f.keys()
                    if isinstance(arrow_refhas_base_f[k], float)
                }
            },
            "by_column_ref_has_arrows_only": {
                "base": arrow_bycol_refhas_base_f,
                "lora": arrow_bycol_refhas_lora_f,
            }
        }
    }

    compare_json_path = os.path.join(CFG["out_dir"], "compare_base_vs_lora_test_with_arrows.json")
    with open(compare_json_path, "w", encoding="utf-8") as f:
        json.dump(out_compare, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {compare_json_path}")

    # Optionally dump all sample details
    if CFG["dump_all_samples_jsonl"]:
        all_path = os.path.join(CFG["out_dir"], "all_samples_with_arrows.jsonl")
        write_jsonl(all_path, sample_results)
        print(f"[OK] wrote {all_path}")

    # -------------------------
    # Bar charts: ONLY samples where ref has arrows (by column: picture/action_memo/dialogue)
    # -------------------------
    # We use the finalized per-column ref-has metrics.
    # Choose metrics that directly support "矢印が取れるようになった".
    metrics_to_plot = [
        ("arrow_presence_recall_if_ref_has", "Arrow Presence Recall", "arrow_bar_presence_recall_refhas.png"),
        ("arrow_count_exact_if_ref_has", "Arrow Count Exact", "arrow_bar_count_exact_refhas.png"),
        ("arrow_direction_f1_micro", "Arrow Direction F1 (micro)", "arrow_bar_direction_f1_refhas.png"),
        ("arrow_seq_exact_if_ref_has", "Arrow Sequence Exact", "arrow_bar_seq_exact_refhas.png"),
        ("arrow_seq_similarity_mean_0_1", "Arrow Sequence Similarity (0-1)", "arrow_bar_seq_similarity_refhas.png"),
    ]

    for key, title, filename in metrics_to_plot:
        base_vals = [float(arrow_bycol_refhas_base_f[c].get(key, 0.0)) for c in arrow_cols]
        lora_vals = [float(arrow_bycol_refhas_lora_f[c].get(key, 0.0)) for c in arrow_cols]
        plot_bar_base_vs_lora(
            columns_order=arrow_cols,
            base_vals=base_vals,
            lora_vals=lora_vals,
            ylabel=key,
            title=f"{title} by column (ref has arrows only): base vs lora",
            out_path=os.path.join(CFG["out_dir"], filename),
        )

    print(f"[OK] wrote arrow bar charts under {CFG['out_dir']}")

    # -------------------------
    # Top-N arrow error examples (miss / direction mismatch / over-detect)
    # + also "LoRA-new errors": base ok but lora fails
    # -------------------------
    def sort_key_miss(rec):
        return (rec["ref_arrow_count"],)  # larger ref arrows => more severe

    def sort_key_dir_mismatch(rec):
        # number of predicted arrows that don't match direction approx = hyp_cnt - tp
        return (rec["lora_arrow_count"] - rec["arrow_tp_lora"], rec["ref_arrow_count"])

    def sort_key_over(rec):
        return (rec["lora_arrow_count"] - rec["ref_arrow_count"], rec["ref_arrow_count"])

    # Restrict example extraction to arrow_columns (picture/action_memo/dialogue) to match your purpose
    samples_arrow_cols = [s for s in sample_results if s["column"] in set(arrow_cols)]

    # lora error sets
    lora_miss = [s for s in samples_arrow_cols if s["arrow_error_lora"] == "miss"]
    lora_dir = [s for s in samples_arrow_cols if s["arrow_error_lora"] == "direction_mismatch"]
    lora_over = [s for s in samples_arrow_cols if s["arrow_error_lora"] == "over_detect"]

    lora_miss_sorted = sorted(lora_miss, key=sort_key_miss, reverse=True)[: CFG["top_n"]]
    lora_dir_sorted = sorted(lora_dir, key=sort_key_dir_mismatch, reverse=True)[: CFG["top_n"]]
    lora_over_sorted = sorted(lora_over, key=sort_key_over, reverse=True)[: CFG["top_n"]]

    # base error sets (for comparison)
    base_miss = [s for s in samples_arrow_cols if s["arrow_error_base"] == "miss"]
    base_dir = [s for s in samples_arrow_cols if s["arrow_error_base"] == "direction_mismatch"]
    base_over = [s for s in samples_arrow_cols if s["arrow_error_base"] == "over_detect"]

    base_miss_sorted = sorted(base_miss, key=lambda s: (s["ref_arrow_count"],), reverse=True)[: CFG["top_n"]]
    base_dir_sorted = sorted(base_dir, key=lambda s: (s["base_arrow_count"] - s["arrow_tp_base"], s["ref_arrow_count"]), reverse=True)[: CFG["top_n"]]
    base_over_sorted = sorted(base_over, key=lambda s: (s["base_arrow_count"] - s["ref_arrow_count"], s["ref_arrow_count"]), reverse=True)[: CFG["top_n"]]

    # LoRA-new errors: base OK but lora error
    lora_new_miss = [s for s in samples_arrow_cols if (s["arrow_base_ok_strict"] == 1 and s["arrow_error_lora"] == "miss")]
    lora_new_dir = [s for s in samples_arrow_cols if (s["arrow_base_ok_strict"] == 1 and s["arrow_error_lora"] == "direction_mismatch")]
    lora_new_over = [s for s in samples_arrow_cols if (s["arrow_base_ok_strict"] == 1 and s["arrow_error_lora"] == "over_detect")]

    lora_new_miss_sorted = sorted(lora_new_miss, key=sort_key_miss, reverse=True)[: CFG["top_n"]]
    lora_new_dir_sorted = sorted(lora_new_dir, key=sort_key_dir_mismatch, reverse=True)[: CFG["top_n"]]
    lora_new_over_sorted = sorted(lora_new_over, key=sort_key_over, reverse=True)[: CFG["top_n"]]

    # write jsonl
    paths = {
        "lora_miss": os.path.join(CFG["out_dir"], "arrow_errors_lora_miss_topN.jsonl"),
        "lora_direction_mismatch": os.path.join(CFG["out_dir"], "arrow_errors_lora_direction_mismatch_topN.jsonl"),
        "lora_over_detect": os.path.join(CFG["out_dir"], "arrow_errors_lora_over_detect_topN.jsonl"),

        "base_miss": os.path.join(CFG["out_dir"], "arrow_errors_base_miss_topN.jsonl"),
        "base_direction_mismatch": os.path.join(CFG["out_dir"], "arrow_errors_base_direction_mismatch_topN.jsonl"),
        "base_over_detect": os.path.join(CFG["out_dir"], "arrow_errors_base_over_detect_topN.jsonl"),

        "lora_new_miss": os.path.join(CFG["out_dir"], "arrow_errors_lora_new_miss_topN.jsonl"),
        "lora_new_direction_mismatch": os.path.join(CFG["out_dir"], "arrow_errors_lora_new_direction_mismatch_topN.jsonl"),
        "lora_new_over_detect": os.path.join(CFG["out_dir"], "arrow_errors_lora_new_over_detect_topN.jsonl"),
    }

    write_jsonl(paths["lora_miss"], lora_miss_sorted)
    write_jsonl(paths["lora_direction_mismatch"], lora_dir_sorted)
    write_jsonl(paths["lora_over_detect"], lora_over_sorted)

    write_jsonl(paths["base_miss"], base_miss_sorted)
    write_jsonl(paths["base_direction_mismatch"], base_dir_sorted)
    write_jsonl(paths["base_over_detect"], base_over_sorted)

    write_jsonl(paths["lora_new_miss"], lora_new_miss_sorted)
    write_jsonl(paths["lora_new_direction_mismatch"], lora_new_dir_sorted)
    write_jsonl(paths["lora_new_over_detect"], lora_new_over_sorted)

    for k, p in paths.items():
        print(f"[OK] wrote {p}")

    # Optional: copy images for quick inspection
    if CFG["copy_images"]:
        def copy_group(group: List[Dict[str, Any]], subdir: str):
            dst = os.path.join(CFG["out_dir"], subdir)
            os.makedirs(dst, exist_ok=True)
            for i, ex in enumerate(group, 1):
                name = f"{i:03d}_cut{ex['cut']}_p{ex['page']}_r{ex['row']}_{ex['column']}.png"
                maybe_copy_image(ex["image_path"], dst, name)

        copy_group(lora_miss_sorted, "arrow_images_lora_miss")
        copy_group(lora_dir_sorted, "arrow_images_lora_direction_mismatch")
        copy_group(lora_over_sorted, "arrow_images_lora_over_detect")

        copy_group(lora_new_miss_sorted, "arrow_images_lora_new_miss")
        copy_group(lora_new_dir_sorted, "arrow_images_lora_new_direction_mismatch")
        copy_group(lora_new_over_sorted, "arrow_images_lora_new_over_detect")

        print(f"[OK] copied arrow images under {CFG['out_dir']}")

    # Print short summary
    print("TEXT TOTAL base:", base_total_f)
    print("TEXT TOTAL lora:", lora_total_f)
    print("ARROW (ref_has only) base:", arrow_refhas_base_f)
    print("ARROW (ref_has only) lora:", arrow_refhas_lora_f)


if __name__ == "__main__":
    main()

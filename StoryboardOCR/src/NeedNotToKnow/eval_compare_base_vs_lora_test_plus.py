# eval_compare_base_vs_lora_test_plus.py
# ------------------------------------------------------------
# Compare BASE vs LoRA on the SAME test set (split_by_cut.json),
# compute CER / ExactMatch (space-insensitive, neutral symbols removed),
# AND additionally:
# 1) Column-wise bar charts (BASE vs LoRA) for CER and ExactMatch
# 2) Extract top-N improved / worsened examples (by CER delta)
#    - Save JSONL with: id, cut/page/row/column, image_path, ref, hyp_base, hyp_lora,
#      cer_base, cer_lora, delta, exact_base, exact_lora
#    - Optionally copy images to output folders for quick inspection (OFF by default)
#
# NOTE: matplotlib colors are not explicitly set (per instruction).
# ------------------------------------------------------------

import os
import csv
import json
import shutil
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
    "lora_adapter_dir": "outputs_ft/stage1_ocr_lora_handwriting_focus",

    # CSV with final_gt (and cut/page/row/column)
    "csv_path": "data/episode01/annotation_table.csv",
    "episode_dir": "data/episode01",

    # split_by_cut.json from training output_dir (to ensure SAME test set)
    "split_by_cut_json": "outputs_ft/stage1_ocr_lora_handwriting_focus/split_by_cut.json",

    # Neutral symbols removed before metrics (space-insensitive)
    "neutral_symbols": ["□", "△"],

    # Evaluate only these columns (None => all)
    "eval_columns": {"cut", "time", "action_memo", "dialogue", "picture"},

    # Generation settings (deterministic)
    "gen": {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    },

    # Prompt (same style as training handwriting-focus code)
    "use_column_conditioned_prompt": True,

    # Limit number of test samples (None => all)
    "max_samples": None,

    # Output base dir (all outputs placed here)
    "out_dir": "outputs_ft/stage1_ocr_lora_handwriting_focus/eval_compare",

    # Top-N examples to export
    "top_n": 30,

    # If True, copy images into output folders for quick review (optional)
    "copy_images": False,
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
# Prompt (same idea as training handwriting-focus)
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
# Data selection: same test cuts
# -------------------------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_test_rows(rows: List[Dict[str, str]], test_cut_ids: List[str]) -> List[Dict[str, str]]:
    test_set = set(test_cut_ids)
    out = []
    for r in rows:
        if r.get("cut", "") not in test_set:
            continue
        if CFG["eval_columns"] and r.get("column", "") not in CFG["eval_columns"]:
            continue
        gt = (r.get("final_gt") or "").strip()
        if not gt:
            continue
        if not r.get("page") or not r.get("row"):
            continue
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
    batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

    out_ids = model.generate(**batch, **CFG["gen"])
    trimmed = out_ids[0][len(batch["input_ids"][0]):]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()
    return text


# -------------------------
# Aggregation helpers
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
    """
    Grouped bar chart: base vs lora per column, with numeric labels.
    No explicit colors are set (matplotlib default).
    """
    x = list(range(len(columns_order)))
    width = 0.35

    plt.figure(figsize=(10, 4))
    b1 = plt.bar([i - width/2 for i in x], base_vals, width, label="base")
    b2 = plt.bar([i + width/2 for i in x], lora_vals, width, label="lora")

    plt.xticks(x, columns_order)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # numeric labels
    ax = plt.gca()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2.0,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.savefig(out_path)
    plt.close()


# -------------------------
# Example export
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
    test_cut_ids = split.get("test_cut_ids", [])
    if not test_cut_ids:
        raise RuntimeError("test_cut_ids is empty in split_by_cut.json. Need enough labeled cuts.")

    rows_all = read_csv_rows(CFG["csv_path"])
    test_rows = filter_test_rows(rows_all, test_cut_ids)
    if CFG["max_samples"] is not None:
        test_rows = test_rows[: int(CFG["max_samples"])]

    if not test_rows:
        raise RuntimeError("No test rows selected. Check CSV and split_by_cut.json.")

    # Load models
    base_model, base_proc = load_base_model()
    lora_model, lora_proc = load_lora_model()

    # Prepare aggregations
    base_total = agg_init()
    lora_total = agg_init()
    base_by_col: Dict[str, Any] = {}
    lora_by_col: Dict[str, Any] = {}

    # Collect per-sample results for top-N extraction
    sample_results: List[Dict[str, Any]] = []

    for idx, r in enumerate(tqdm(test_rows, desc="Eval test: base vs lora")):
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

        # metrics
        cer_b = cer(hyp_base, ref, CFG["neutral_symbols"])
        cer_l = cer(hyp_lora, ref, CFG["neutral_symbols"])
        em_b = exact_match(hyp_base, ref, CFG["neutral_symbols"])
        em_l = exact_match(hyp_lora, ref, CFG["neutral_symbols"])
        delta = cer_l - cer_b

        # aggregate
        update_agg(base_total, hyp_base, ref, CFG["neutral_symbols"])
        update_agg(lora_total, hyp_lora, ref, CFG["neutral_symbols"])

        if col not in base_by_col:
            base_by_col[col] = agg_init()
            lora_by_col[col] = agg_init()
        update_agg(base_by_col[col], hyp_base, ref, CFG["neutral_symbols"])
        update_agg(lora_by_col[col], hyp_lora, ref, CFG["neutral_symbols"])

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
            "delta_cer_lora_minus_base": delta,
            "exact_base": em_b,
            "exact_lora": em_l,
        })

    # Finalize totals
    base_total_f = finalize_agg(base_total)
    lora_total_f = finalize_agg(lora_total)

    base_cols_f = {k: finalize_agg(v) for k, v in base_by_col.items()}
    lora_cols_f = {k: finalize_agg(v) for k, v in lora_by_col.items()}

    delta_total = {
        "cer": lora_total_f["cer"] - base_total_f["cer"],
        "exact_match": lora_total_f["exact_match"] - base_total_f["exact_match"],
    }

    # Save main comparison json
    out_compare = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model_id": CFG["base_model_id"],
            "lora_adapter_dir": CFG["lora_adapter_dir"],
            "csv_path": CFG["csv_path"],
            "split_by_cut_json": CFG["split_by_cut_json"],
            "test_cut_ids_count": len(test_cut_ids),
            "test_samples_used": base_total_f["n"],
            "neutral_symbols": CFG["neutral_symbols"],
            "gen": CFG["gen"],
        },
        "total": {
            "base": base_total_f,
            "lora": lora_total_f,
            "delta_lora_minus_base": delta_total,
        },
        "by_column": {
            "base": base_cols_f,
            "lora": lora_cols_f,
        }
    }

    compare_json_path = os.path.join(CFG["out_dir"], "compare_base_vs_lora_test.json")
    with open(compare_json_path, "w", encoding="utf-8") as f:
        json.dump(out_compare, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {compare_json_path}")

    # -------------------------
    # 1) Column-wise bar charts (base vs lora)
    # -------------------------
    # Use stable column order
    col_order = ["cut", "time", "action_memo", "dialogue", "picture"]
    if CFG["eval_columns"]:
        # keep only those evaluated, preserve order
        col_order = [c for c in col_order if c in CFG["eval_columns"]]

    base_cer_vals = []
    lora_cer_vals = []
    base_em_vals = []
    lora_em_vals = []

    for c in col_order:
        base_cer_vals.append(float(base_cols_f.get(c, {}).get("cer", 0.0)))
        lora_cer_vals.append(float(lora_cols_f.get(c, {}).get("cer", 0.0)))
        base_em_vals.append(float(base_cols_f.get(c, {}).get("exact_match", 0.0)))
        lora_em_vals.append(float(lora_cols_f.get(c, {}).get("exact_match", 0.0)))

    plot_bar_base_vs_lora(
        columns_order=col_order,
        base_vals=base_cer_vals,
        lora_vals=lora_cer_vals,
        ylabel="CER (lower is better)",
        title="CER by column (base vs lora) on SAME test set",
        out_path=os.path.join(CFG["out_dir"], "bar_cer_base_vs_lora.png"),
    )
    plot_bar_base_vs_lora(
        columns_order=col_order,
        base_vals=base_em_vals,
        lora_vals=lora_em_vals,
        ylabel="Exact Match (higher is better)",
        title="Exact Match by column (base vs lora) on SAME test set",
        out_path=os.path.join(CFG["out_dir"], "bar_exact_match_base_vs_lora.png"),
    )
    print(f"[OK] wrote bar charts under {CFG['out_dir']}")

    # -------------------------
    # 2) Top-N improved/worsened examples
    # -------------------------
    # Improved: delta_cer (lora-base) is negative (more negative is better)
    sample_results_sorted = sorted(sample_results, key=lambda x: x["delta_cer_lora_minus_base"])
    top_improved = sample_results_sorted[: CFG["top_n"]]

    # Worsened: most positive deltas
    top_worsened = list(reversed(sample_results_sorted[-CFG["top_n"]:]))

    improved_path = os.path.join(CFG["out_dir"], "top_improved.jsonl")
    worsened_path = os.path.join(CFG["out_dir"], "top_worsened.jsonl")
    write_jsonl(improved_path, top_improved)
    write_jsonl(worsened_path, top_worsened)
    print(f"[OK] wrote {improved_path}")
    print(f"[OK] wrote {worsened_path}")

    # Optional: copy images
    if CFG["copy_images"]:
        imp_dir = os.path.join(CFG["out_dir"], "top_improved_images")
        wor_dir = os.path.join(CFG["out_dir"], "top_worsened_images")
        for i, ex in enumerate(top_improved, 1):
            name = f"{i:03d}_cut{ex['cut']}_p{ex['page']}_r{ex['row']}_{ex['column']}.png"
            maybe_copy_image(ex["image_path"], imp_dir, name)
        for i, ex in enumerate(top_worsened, 1):
            name = f"{i:03d}_cut{ex['cut']}_p{ex['page']}_r{ex['row']}_{ex['column']}.png"
            maybe_copy_image(ex["image_path"], wor_dir, name)
        print(f"[OK] copied images under {CFG['out_dir']}")

    # Print summary to console
    print("TOTAL base:", base_total_f)
    print("TOTAL lora:", lora_total_f)
    print("DELTA (lora-base):", delta_total)


if __name__ == "__main__":
    main()

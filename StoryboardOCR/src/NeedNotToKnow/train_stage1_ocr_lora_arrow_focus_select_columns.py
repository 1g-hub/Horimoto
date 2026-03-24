# train_stage1_ocr_lora_arrow_focus_select_columns.py
# ------------------------------------------------------------
# 목적:
# - 矢印(→←↑↓)の手書き認識を改善するため、
#   1) 通常の列OCRデータ(annotation_table.csv)
#   2) 矢印切り抜きデータ(arrow_crops.csv)
#   を混ぜて LoRA 学習する。
# - 学習後、同一testセットで base / LoRA / 列別mix を比較し、
#   「改善した列だけLoRA採用」する column_model_map.json を作る。
#
# 出力:
# - output_dir/
#   - adapter/           (LoRA adapter)
#   - split_by_cut.json  (train/val/test cut split)
#   - eval_test_compare.json (base vs lora vs mixed)
#   - column_model_map.json  (column -> base/lora)
#   - bar_text_cer.png
#   - bar_arrow_seq_edit_rate_refhas.png
# ------------------------------------------------------------

import os
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps, ImageEnhance

from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model

# Optional (if you want to load adapter for evaluation as PeftModel)
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# =========================
# CONFIG
# =========================
CFG = {
    # Base model (4B/8B などに変更)
    "base_model_id": "Qwen/Qwen3-VL-4B-Instruct",

    # Data
    "episode_dir": "data/episode01",
    "annotation_csv": "data/episode01/annotation_table.csv",

    # Arrow crop dataset (optional). 空なら切り抜き無しで動く。
    "arrow_crops_csv": "data/arrow_crops_133.csv",
    "use_arrow_crops": True,

    # Use only these columns from annotation_table
    "train_columns": {"picture", "action_memo", "dialogue", "cut", "time"},
    "eval_columns": {"picture", "action_memo", "dialogue", "cut", "time"},

    # Range filtering (optional)
    "select_cuts": True,
    "cut_start": 1,
    "cut_end": 255,

    "select_pages": False,
    "page_start": 1,
    "page_end": 9999,

    # Split
    "seed": 42,
    "val_ratio": 0.1,
    "test_ratio": 0.8,

    # Output
    "output_dir": "outputs_ft/stage1_ocr_lora_133arrow_focus_cut255_train10per",
    "save_adapter_dirname": "adapter",

    # Training
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,

    "logging_steps": 1,
    "save_strategy": "epoch",
    "evaluation_strategy": "steps",
    "save_steps": 300,
    "save_total_limit": 2,

    "bf16": True,
    "fp16": False,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # LoRA target preference:
    #   "vision" : vision projector / multimodal fusion 寄りを狙う（見つからなければ text fallback）
    #   "text"   : q_proj/k_proj/... を狙う
    #   "both"   : 両方
    "lora_target_mode": "vision",

    # Weighted sampling (重要)
    "use_weighted_sampling": True,
    # 通常サンプルの重み
    "base_weight": 1.0,
    # 正解に矢印が含まれる通常サンプルの重み倍率
    "arrow_in_gt_weight_mul": 2.0,
    # 矢印切り抜きサンプルの重み倍率
    "arrow_crop_weight_mul": 3.0,

    # Augmentation (手書き矢印を強くする)
    "use_augmentation": True,
    "aug_prob": 0.7,
    "aug_autocontrast_prob": 0.4,
    "aug_contrast_prob": 0.6,
    "aug_contrast_min": 1.8,
    "aug_contrast_max": 3.0,
    "aug_sharpen_prob": 0.3,
    "aug_sharpen_min": 1.2,
    "aug_sharpen_max": 1.8,
    "aug_small_rotate_prob": 0.3,
    "aug_rotate_deg": 2.0,

    # Generation for evaluation
    "gen": {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    },

    # Metrics normalization
    "neutral_symbols": ["□", "△"],
    "space_insensitive": True,

    # Column selection policy (列別mix)
    # cut/time は壊れやすいので base に固定したい場合ここに入れる
    "force_base_columns": {"cut", "time"},
    # LoRA採用判定: CERがこれ以上改善したら採用（負なので -0.01 など）
    "min_cer_improve_to_adopt": 0.01,  # 0.01 = CERが 0.01 以上下がったら採用
    # 矢印指標重視列
    "arrow_sensitive_columns": {"picture", "action_memo", "dialogue"},

    # Loss plot
    "loss_plot_filename": "loss_curve_train_val.png",
    "log_history_filename": "log_history.json",
    "split_filename": "split_by_cut.json",
}
# =========================


ARROWS = ["→", "←", "↑", "↓"]


# -------------------------
# small utils
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def in_cut_range(cut: Optional[int]) -> bool:
    if not CFG["select_cuts"]:
        return True
    if cut is None:
        return False
    return CFG["cut_start"] <= cut <= CFG["cut_end"]


def in_page_range(page: Optional[int]) -> bool:
    if not CFG["select_pages"]:
        return True
    if page is None:
        return False
    return CFG["page_start"] <= page <= CFG["page_end"]


def has_arrow(s: str) -> bool:
    if not s:
        return False
    return any(a in s for a in ARROWS)


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
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (0 if a == b else 1)
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


def extract_arrow_seq(s: str) -> List[str]:
    s = s or ""
    return [ch for ch in s if ch in ARROWS]


def arrow_metrics_from_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    pairs: list of (ref, hyp)
    指標は「refに矢印があるサンプルのみ」で計算する想定。
    """
    # filter ref_has
    filtered = []
    for ref, hyp in pairs:
        ref_seq = extract_arrow_seq(ref)
        if len(ref_seq) > 0:
            filtered.append((ref, hyp))

    n = len(filtered)
    out = {
        "n_samples": float(n),
        "n_ref_has_arrow": float(n),
    }
    if n == 0:
        # no arrow refs
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

    # presence/count/seq
    presence_hits = 0
    count_exact = 0
    count_abs_err_sum = 0.0
    seq_exact = 0

    # direction micro (multiset)
    tp = 0
    fp = 0
    fn = 0

    # seq edit micro
    total_edit = 0
    total_ref_len = 0
    sim_sum = 0.0

    for ref, hyp in filtered:
        rseq = extract_arrow_seq(ref)
        hseq = extract_arrow_seq(hyp)

        if len(hseq) > 0:
            presence_hits += 1

        if len(hseq) == len(rseq):
            count_exact += 1
        count_abs_err_sum += abs(len(hseq) - len(rseq))

        if hseq == rseq:
            seq_exact += 1

        # direction micro as multiset
        from collections import Counter
        rc = Counter(rseq)
        hc = Counter(hseq)
        for a in ARROWS:
            tp += min(rc.get(a, 0), hc.get(a, 0))
            fp += max(hc.get(a, 0) - rc.get(a, 0), 0)
            fn += max(rc.get(a, 0) - hc.get(a, 0), 0)

        # seq edit
        edit = levenshtein(hseq, rseq)
        total_edit += edit
        total_ref_len += len(rseq)
        sim = 1.0 - (edit / len(rseq)) if len(rseq) > 0 else (1.0 if len(hseq) == 0 else 0.0)
        sim_sum += sim

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
# prompts (列別)
# -------------------------
BASE_RULES = (
    "Rules:\n"
    "- Keep original line breaks.\n"
    "- Preserve symbols exactly (→ ← ↑ ↓, brackets, punctuation).\n"
    "- Do NOT correct typos.\n"
    "- Do NOT translate.\n"
    "- If unreadable, output '□'.\n"
    "- Return ONLY the transcribed text. No explanations.\n"
)

def build_prompt_for_column(col: str, is_arrow_crop: bool) -> str:
    if is_arrow_crop:
        return (
            "You are an OCR engine for anime storyboards.\n"
            "This is a SMALL CROP focusing on handwritten arrows/symbols.\n"
            "Task: Transcribe ALL visible text/symbols faithfully.\n"
            "Arrows (→ ← ↑ ↓) are especially important.\n\n"
            f"{BASE_RULES}\n"
        )

    if col == "cut":
        return (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: CUT.\n"
            "Task: Read the cut number and/or time if present.\n"
            "Handwritten digits may appear.\n\n"
            f"{BASE_RULES}\n"
        )
    if col == "time":
        return (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: TIME.\n"
            "Task: Transcribe timing expressions (mm:ss, m+n, frame counts).\n\n"
            f"{BASE_RULES}\n"
        )
    if col == "picture":
        return (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: PICTURE.\n"
            "Most of this region is drawings.\n"
            "Task: Transcribe ONLY the text/symbols if they exist.\n"
            "Handwritten arrows and short terms are common.\n"
            "If there is no text, output '□'.\n\n"
            f"{BASE_RULES}\n"
        )
    if col == "action_memo":
        return (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: ACTION MEMO.\n"
            "Task: Transcribe action/camera/timing notes.\n"
            "Production terms and arrows are common.\n\n"
            f"{BASE_RULES}\n"
        )
    if col == "dialogue":
        return (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: DIALOGUE.\n"
            "Task: Transcribe dialogue and audio notes.\n"
            "Speaker names, quotes 「」, SE/MA/BGM, arrows may appear.\n\n"
            f"{BASE_RULES}\n"
        )
    return "You are an OCR engine.\n\n" + BASE_RULES


# -------------------------
# image augmentation (PIL)
# -------------------------
def maybe_augment(img: Image.Image) -> Image.Image:
    if not CFG["use_augmentation"]:
        return img
    if random.random() > CFG["aug_prob"]:
        return img

    x = img

    # autocontrast
    if random.random() < CFG["aug_autocontrast_prob"]:
        x = ImageOps.autocontrast(x)

    # contrast boost (binarize-ish)
    if random.random() < CFG["aug_contrast_prob"]:
        factor = random.uniform(CFG["aug_contrast_min"], CFG["aug_contrast_max"])
        x = ImageEnhance.Contrast(x).enhance(factor)

    # sharpen
    if random.random() < CFG["aug_sharpen_prob"]:
        factor = random.uniform(CFG["aug_sharpen_min"], CFG["aug_sharpen_max"])
        x = ImageEnhance.Sharpness(x).enhance(factor)

    # small rotation
    if random.random() < CFG["aug_small_rotate_prob"]:
        deg = random.uniform(-CFG["aug_rotate_deg"], CFG["aug_rotate_deg"])
        x = x.rotate(deg, resample=Image.BICUBIC, expand=False)

    return x


# -------------------------
# csv loading
# -------------------------
def read_csv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def reconstruct_image_path_from_annotation(row: Dict[str, str]) -> str:
    col = row["column"]
    page = row["page"]
    r = row["row"]
    return os.path.join(CFG["episode_dir"], col, f"page{page}_row{r}.png")


def load_annotation_samples() -> List[Dict[str, Any]]:
    rows = read_csv(CFG["annotation_csv"])
    out = []
    for r in rows:
        col = (r.get("column") or "").strip()
        if col not in CFG["train_columns"]:
            continue

        cut_i = safe_int(r.get("cut"))
        page_i = safe_int(r.get("page"))
        row_i = safe_int(r.get("row"))
        if not in_cut_range(cut_i):
            continue
        if not in_page_range(page_i):
            continue

        gt = (r.get("final_gt") or "")
        if not gt.strip():
            continue

        img_path = reconstruct_image_path_from_annotation(r)
        if not os.path.isfile(img_path):
            continue

        out.append({
            "sample_type": "main",
            "cut": str(cut_i) if cut_i is not None else "",
            "page": str(page_i) if page_i is not None else "",
            "row": str(row_i) if row_i is not None else "",
            "column": col,
            "image_path": img_path,
            "final_gt": gt,
            "has_arrow": has_arrow(gt),
        })
    return out


def load_arrow_crop_samples() -> List[Dict[str, Any]]:
    if not CFG["use_arrow_crops"]:
        return []
    rows = read_csv(CFG["arrow_crops_csv"])
    out = []
    for r in rows:
        id = (r.get("id") or "").strip()
        if not id:
            continue
        img_path = "data/arrows/" + id + ".png"
        if not os.path.isabs(img_path):
            # relative path is interpreted from project root
            img_path = os.path.normpath(img_path)

        if not os.path.isfile(img_path):
            print("NOT DATA:::::::::::::")
            continue

        gt = (r.get("final_gt") or "")
        if not gt.strip():
            continue

        col = (r.get("column") or "").strip()
        if not col:
            col = "picture"  # default

        cut_i = safe_int(r.get("cut"))
        page_i = safe_int(r.get("page"))
        row_i = safe_int(r.get("row"))

        # arrow crops are typically for arrows, but we don't force it
        out.append({
            "sample_type": "arrow_crop",
            "cut": str(cut_i) if cut_i is not None else "",
            "page": str(page_i) if page_i is not None else "",
            "row": str(row_i) if row_i is not None else "",
            "column": col,
            "image_path": img_path,
            "final_gt": gt,
            "has_arrow": has_arrow(gt),
        })
    return out


# -------------------------
# split by cut (no leakage)
# -------------------------
def split_by_cut(samples: List[Dict[str, Any]], seed: int, val_ratio: float, test_ratio: float) -> Tuple[List, List, List, Dict[str, Any]]:
    # cut が空のサンプルは train に落とす（arrow_crop で cut が無い場合など）
    cut_ids = sorted({s["cut"] for s in samples if s.get("cut")})

    rnd = random.Random(seed)
    rnd.shuffle(cut_ids)

    n = len(cut_ids)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    if n >= 10:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    if n_val + n_test >= n:
        n_val = max(0, n // 10)
        n_test = max(0, n // 10)

    val_set = set(cut_ids[:n_val])
    test_set = set(cut_ids[n_val:n_val + n_test])
    train_set = set(cut_ids[n_val + n_test:])

    train, val, test = [], [], []
    for s in samples:
        c = s.get("cut", "")
        if not c:
            train.append(s)
        elif c in val_set:
            val.append(s)
        elif c in test_set:
            test.append(s)
        else:
            train.append(s)

    info = {
        "train_cut_ids": sorted(list(train_set)),
        "val_cut_ids": sorted(list(val_set)),
        "test_cut_ids": sorted(list(test_set)),
        "n_cuts": n,
    }
    return train, val, test, info


# -------------------------
# Dataset + collator
# -------------------------
class MixedOCRDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], processor: AutoProcessor, is_train: bool):
        self.samples = samples
        self.processor = processor
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.is_train:
            img = maybe_augment(img)

        is_arrow_crop = (s["sample_type"] == "arrow_crop")
        prompt = build_prompt_for_column(s["column"], is_arrow_crop=is_arrow_crop)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]

        prompt_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        tok = self.processor.tokenizer
        gt = s["final_gt"]
        tgt_ids = tok(gt, add_special_tokens=False).input_ids
        if tok.eos_token_id is not None:
            tgt_ids = tgt_ids + [tok.eos_token_id]

        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_mask = prompt_inputs["attention_mask"][0]
        tgt_t = torch.tensor(tgt_ids, dtype=prompt_ids.dtype)

        input_ids = torch.cat([prompt_ids, tgt_t], dim=0)
        attention_mask = torch.cat([prompt_mask, torch.ones_like(tgt_t)], dim=0)

        labels = torch.full_like(input_ids, fill_value=-100)
        labels[len(prompt_ids):] = tgt_t

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,

            # for evaluation/analysis
            "meta_cut": s.get("cut", ""),
            "meta_page": s.get("page", ""),
            "meta_row": s.get("row", ""),
            "meta_column": s.get("column", ""),
            "meta_image_path": s.get("image_path", ""),
            "meta_ref": gt,
        }

        # vision tensors
        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = v[0]
        return out


@dataclass
class DataCollatorForVL:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tok = self.processor.tokenizer

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = tok.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), fill_value=-100, dtype=batch["input_ids"].dtype)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        batch["labels"] = padded_labels

        # stack vision keys only
        for k in features[0].keys():
            if k in (
                "input_ids", "attention_mask", "labels",
                "meta_cut", "meta_page", "meta_row", "meta_column", "meta_image_path", "meta_ref"
            ):
                continue
            vals = [f[k] for f in features]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals, dim=0)

        # keep meta as list (not passed to model)
        batch["meta_cut"] = [f["meta_cut"] for f in features]
        batch["meta_page"] = [f["meta_page"] for f in features]
        batch["meta_row"] = [f["meta_row"] for f in features]
        batch["meta_column"] = [f["meta_column"] for f in features]
        batch["meta_image_path"] = [f["meta_image_path"] for f in features]
        batch["meta_ref"] = [f["meta_ref"] for f in features]

        return batch


# -------------------------
# Weighted sampling
# -------------------------
def build_sample_weights(samples: List[Dict[str, Any]]) -> List[float]:
    wts = []
    for s in samples:
        w = float(CFG["base_weight"])
        if s.get("has_arrow"):
            w *= float(CFG["arrow_in_gt_weight_mul"])
        if s.get("sample_type") == "arrow_crop":
            w *= float(CFG["arrow_crop_weight_mul"])
        wts.append(w)
    return wts


class WeightedTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if not CFG["use_weighted_sampling"]:
            return super().get_train_dataloader()
        ds = self.train_dataset
        samples = getattr(ds, "samples", None)
        if samples is None:
            return super().get_train_dataloader()
        weights = build_sample_weights(samples)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            ds,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )


# -------------------------
# LoRA target selection
# -------------------------
def infer_lora_targets(model: nn.Module, mode: str) -> List[str]:
    """
    PEFT target_modules は「名前の部分一致」で刺さる。
    - vision: 視覚/マルチモーダル結合っぽい層(Linear)の full name を拾う
    - text  : q_proj/k_proj/... の leaf 名
    - both  : union
    """
    text_leaf = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    vision_name_keywords = [
        "vision", "visual", "mm", "projector", "proj", "merger", "resampler", "connector"
    ]

    vision_fullnames = []
    found_text_leaf = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in text_leaf:
                found_text_leaf.add(leaf)

            lname = name.lower()
            if any(k in lname for k in vision_name_keywords):
                # full name as target for precision
                vision_fullnames.append(name)

    # de-dup
    vision_fullnames = sorted(list(dict.fromkeys(vision_fullnames)))

    if mode == "vision":
        # If nothing found, fallback to text leaf
        return vision_fullnames if len(vision_fullnames) > 0 else sorted(list(found_text_leaf or text_leaf))
    if mode == "text":
        return sorted(list(found_text_leaf or text_leaf))
    # both
    targets = set(vision_fullnames) | set(found_text_leaf or text_leaf)
    return sorted(list(targets))


# -------------------------
# generation helper (safe)
# -------------------------
@torch.no_grad()
def safe_generate_one(model, processor, img: Image.Image, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
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
    # move to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _run(b):
        out_ids = model.generate(
            **b,
            pad_token_id=pad_id,
            **CFG["gen"],
        )
        prompt_len = b["input_ids"].shape[-1]
        text = tok.decode(out_ids[0][prompt_len:], skip_special_tokens=True).strip()
        return text

    try:
        return _run(batch)
    except IndexError as e:
        # Qwen3-VL generation で attention_mask 次元が原因の例外が出る場合の保険
        msg = str(e)
        if "shape of the mask" in msg and "does not match" in msg and "attention_mask" in batch:
            am = batch["attention_mask"]
            if isinstance(am, torch.Tensor) and am.dim() == 2:
                batch2 = dict(batch)
                batch2["attention_mask"] = am.unsqueeze(1)
                return _run(batch2)
        raise


# -------------------------
# evaluation on samples (base/lora/mixed)
# -------------------------
def make_key(s: Dict[str, Any]) -> str:
    return f"cut={s.get('cut','')}/page={s.get('page','')}/row={s.get('row','')}/col={s.get('column','')}/img={s.get('image_path','')}"

@torch.no_grad()
def run_inference_records(model, processor, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recs = []
    for s in tqdm(samples, desc="Infer", leave=False):
        col = s["column"]
        img = Image.open(s["image_path"]).convert("RGB")
        prompt = build_prompt_for_column(col, is_arrow_crop=False)  # 推論は通常列想定
        hyp = safe_generate_one(model, processor, img, prompt)
        recs.append({
            "key": make_key(s),
            "cut": s.get("cut", ""),
            "page": s.get("page", ""),
            "row": s.get("row", ""),
            "column": col,
            "image_path": s.get("image_path", ""),
            "ref": s.get("final_gt", ""),
            "hyp": hyp,
        })
    return recs


def compute_text_metrics_from_records(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cer = 0.0
    total_em = 0
    n = 0

    by_col: Dict[str, Any] = {}
    for r in recs:
        ref = r["ref"]
        hyp = r["hyp"]
        c = cer(hyp, ref)
        e = exact_match(hyp, ref)
        total_cer += c
        total_em += e
        n += 1
        col = r["column"]
        if col not in by_col:
            by_col[col] = {"n": 0, "cer_sum": 0.0, "em_sum": 0}
        by_col[col]["n"] += 1
        by_col[col]["cer_sum"] += c
        by_col[col]["em_sum"] += e

    out = {
        "n": n,
        "cer": total_cer / max(n, 1),
        "exact_match": total_em / max(n, 1),
        "by_column": {},
    }
    for col, agg in by_col.items():
        out["by_column"][col] = {
            "n": agg["n"],
            "cer": agg["cer_sum"] / max(agg["n"], 1),
            "exact_match": agg["em_sum"] / max(agg["n"], 1),
        }
    return out


def compute_arrow_metrics_bycol(recs: List[Dict[str, Any]], cols: List[str]) -> Dict[str, Any]:
    out = {}
    for col in cols:
        pairs = [(r["ref"], r["hyp"]) for r in recs if r["column"] == col]
        out[col] = arrow_metrics_from_pairs(pairs)
    return out


def plot_bar_base_vs_lora(columns: List[str], base_vals: List[float], lora_vals: List[float], ylabel: str, title: str, out_path: str):
    x = list(range(len(columns)))
    width = 0.35
    plt.figure(figsize=(10, 4))
    b1 = plt.bar([i - width/2 for i in x], base_vals, width, label="base")
    b2 = plt.bar([i + width/2 for i in x], lora_vals, width, label="lora")
    plt.xticks(x, columns)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    ax = plt.gca()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.savefig(out_path)
    plt.close()


def decide_column_map(
    base_text: Dict[str, Any],
    lora_text: Dict[str, Any],
    base_arrow_bycol: Dict[str, Any],
    lora_arrow_bycol: Dict[str, Any],
) -> Dict[str, str]:
    """
    column -> "base" or "lora"
    方針:
      - force_base_columns は常に base
      - それ以外は CER が一定以上改善したら lora
      - 矢印重視列は、矢印 edit_rate が改善しても採用する（ただしCERが極端に悪化はしないこと）
    """
    col_map = {}

    all_cols = sorted(list(CFG["eval_columns"]))
    for col in all_cols:
        if col in CFG["force_base_columns"]:
            col_map[col] = "base"
            continue

        b = base_text["by_column"].get(col, {})
        l = lora_text["by_column"].get(col, {})
        b_cer = float(b.get("cer", 999))
        l_cer = float(l.get("cer", 999))

        cer_improve = (b_cer - l_cer)  # positive is good

        adopt = False

        # CER based
        if cer_improve >= float(CFG["min_cer_improve_to_adopt"]):
            adopt = True

        # arrow based for arrow columns
        if (col in CFG["arrow_sensitive_columns"]):
            b_arrow = base_arrow_bycol.get(col, {})
            l_arrow = lora_arrow_bycol.get(col, {})
            b_edit = float(b_arrow.get("arrow_seq_edit_rate_micro", 999))
            l_edit = float(l_arrow.get("arrow_seq_edit_rate_micro", 999))
            # edit_rate improve
            if (b_edit - l_edit) >= 0.05 and (l_cer <= b_cer + 0.02):
                adopt = True

        col_map[col] = "lora" if adopt else "base"

    return col_map


def compute_mixed_records(base_recs: List[Dict[str, Any]], lora_recs: List[Dict[str, Any]], col_map: Dict[str, str]) -> List[Dict[str, Any]]:
    bmap = {r["key"]: r for r in base_recs}
    lmap = {r["key"]: r for r in lora_recs}
    keys = sorted(set(bmap.keys()) & set(lmap.keys()))
    mixed = []
    for k in keys:
        b = bmap[k]
        l = lmap[k]
        col = b["column"]
        use = col_map.get(col, "base")
        hyp = l["hyp"] if use == "lora" else b["hyp"]
        mixed.append({**b, "hyp": hyp})
    return mixed


def plot_loss_curve(log_history: List[Dict[str, Any]], out_path: str):
    """
    Save a single PNG that includes:
      - train loss curve (key: "loss")
      - val loss curve (key: "eval_loss")
    """
    steps_train, loss_train = [], []
    steps_val, loss_val = [], []

    for item in log_history:
        # Train loss logs typically have "loss" and "step", without "eval_loss"
        if "loss" in item and "step" in item and "eval_loss" not in item:
            steps_train.append(item["step"])
            loss_train.append(item["loss"])

        # Eval loss logs have "eval_loss" and "step"
        if "eval_loss" in item and "step" in item:
            steps_val.append(item["step"])
            loss_val.append(item["eval_loss"])

    if not steps_train and not steps_val:
        print("[WARN] No loss logs found. Increase logging/eval frequency.")
        return

    plt.figure()
    if steps_train:
        plt.plot(steps_train, loss_train, label="train_loss")
    if steps_val:
        plt.plot(steps_val, loss_val, label="val_loss")

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] wrote loss plot: {out_path}")


# -------------------------
# main
# -------------------------
def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    os.makedirs(CFG["output_dir"], exist_ok=True)
    adapter_out_dir = os.path.join(CFG["output_dir"], CFG["save_adapter_dirname"])
    os.makedirs(adapter_out_dir, exist_ok=True)

    # load samples
    main_samples = load_annotation_samples()
    arrow_samples = load_arrow_crop_samples()
    all_samples = main_samples + arrow_samples

    if not all_samples:
        raise RuntimeError("No training samples found. Check CSV paths and image paths.")

    # split by cut
    train_s, val_s, test_s, split_info = split_by_cut(all_samples, CFG["seed"], CFG["val_ratio"], CFG["test_ratio"])

    # save split file (for reproducible eval)
    split_path = os.path.join(CFG["output_dir"], "split_by_cut.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": now_iso(),
            "split_info": split_info,
            "filters": {
                "select_cuts": CFG["select_cuts"],
                "cut_start": CFG["cut_start"],
                "cut_end": CFG["cut_end"],
                "select_pages": CFG["select_pages"],
                "page_start": CFG["page_start"],
                "page_end": CFG["page_end"],
                "train_columns": sorted(list(CFG["train_columns"])),
                "eval_columns": sorted(list(CFG["eval_columns"])),
            },
            "counts": {
                "train_samples": len(train_s),
                "val_samples": len(val_s),
                "test_samples": len(test_s),
                "train_main": sum(1 for x in train_s if x["sample_type"] == "main"),
                "train_arrow_crops": sum(1 for x in train_s if x["sample_type"] == "arrow_crop"),
            }
        }, f, ensure_ascii=False, indent=2)

    # model + processor
    processor = AutoProcessor.from_pretrained(CFG["base_model_id"])
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if CFG["bf16"] else "auto",
    )

    # LoRA config
    targets = infer_lora_targets(base_model, CFG["lora_target_mode"])
    print("[LoRA] target_modules count =", len(targets))
    # 多すぎる場合はログがうるさいので先頭だけ表示
    print("[LoRA] target_modules head =", targets[:20])

    lora_cfg = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # datasets
    train_ds = MixedOCRDataset(train_s, processor, is_train=True)
    val_ds = MixedOCRDataset(val_s, processor, is_train=False) if len(val_s) > 0 else None

    collator = DataCollatorForVL(processor)

    args = TrainingArguments(
        output_dir=CFG["output_dir"],
        num_train_epochs=CFG["num_train_epochs"],
        per_device_train_batch_size=CFG["per_device_train_batch_size"],
        per_device_eval_batch_size=CFG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
        learning_rate=CFG["learning_rate"],
        warmup_ratio=CFG["warmup_ratio"],
        weight_decay=CFG["weight_decay"],
        logging_steps=CFG["logging_steps"],
        save_strategy=CFG["save_strategy"],
        save_steps=CFG["save_steps"],
        eval_strategy=CFG["evaluation_strategy"] if val_ds is not None else "no",
        eval_steps=CFG["save_steps"] if val_ds is not None else None,
        save_total_limit=CFG["save_total_limit"],
        report_to="none",
        remove_unused_columns=False,  # ★VLのキーを落とさない
        bf16=CFG["bf16"],
        fp16=CFG["fp16"],
        load_best_model_at_end=False,  # generation系の指標を metric_for_best_model にするなら拡張が必要。まずは安定優先。
    )

    trainer_cls = WeightedTrainer if CFG["use_weighted_sampling"] else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    # save adapter
    trainer.save_model(adapter_out_dir)
    print(f"[OK] saved adapter to {adapter_out_dir}")

    # -------------------------
    # EVAL: base vs lora vs mixed (on SAME test subset = test_s)
    # 推論は「列画像(通常crop)」で見るのが目的なので、
    # test_s から main サンプルのみを評価対象にする（arrow_cropは評価ノイズになりやすい）
    # -------------------------
    test_main = [s for s in test_s if s["sample_type"] == "main" and s["column"] in CFG["eval_columns"]]
    if len(test_main) == 0:
        print("[WARN] test_main is empty. Evaluation skipped.")
        return

    # load base model again (clean)
    base_eval_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"], device_map="auto", torch_dtype="auto"
    ).eval()
    base_eval_proc = AutoProcessor.from_pretrained(CFG["base_model_id"])

    # load lora model
    if PeftModel is None:
        raise RuntimeError("peft is required to load adapter for evaluation.")
    lora_eval_base = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"], device_map="auto", torch_dtype="auto"
    )
    lora_eval_model = PeftModel.from_pretrained(lora_eval_base, adapter_out_dir).eval()
    lora_eval_proc = AutoProcessor.from_pretrained(CFG["base_model_id"])

    # inference
    base_recs = run_inference_records(base_eval_model, base_eval_proc, test_main)
    lora_recs = run_inference_records(lora_eval_model, lora_eval_proc, test_main)

    # metrics
    base_text = compute_text_metrics_from_records(base_recs)
    lora_text = compute_text_metrics_from_records(lora_recs)

    # arrow metrics by column (ref_has only)
    cols_for_arrow = ["picture", "action_memo", "dialogue"]
    base_arrow_bycol = compute_arrow_metrics_bycol(base_recs, cols_for_arrow)
    lora_arrow_bycol = compute_arrow_metrics_bycol(lora_recs, cols_for_arrow)

    # decide column map
    col_map = decide_column_map(base_text, lora_text, base_arrow_bycol, lora_arrow_bycol)

    # mixed
    mixed_recs = compute_mixed_records(base_recs, lora_recs, col_map)
    mixed_text = compute_text_metrics_from_records(mixed_recs)
    mixed_arrow_bycol = compute_arrow_metrics_bycol(mixed_recs, cols_for_arrow)

    # save compare json
    out_compare = {
        "meta": {
            "created_at": now_iso(),
            "base_model_id": CFG["base_model_id"],
            "adapter_dir": adapter_out_dir,
            "test_main_n": len(test_main),
            "split_by_cut": split_info,
            "force_base_columns": sorted(list(CFG["force_base_columns"])),
            "min_cer_improve_to_adopt": CFG["min_cer_improve_to_adopt"],
        },
        "text_total": {
            "base": {"n": base_text["n"], "cer": base_text["cer"], "exact_match": base_text["exact_match"]},
            "lora": {"n": lora_text["n"], "cer": lora_text["cer"], "exact_match": lora_text["exact_match"]},
            "mixed": {"n": mixed_text["n"], "cer": mixed_text["cer"], "exact_match": mixed_text["exact_match"]},
        },
        "text_by_column": {
            "base": base_text["by_column"],
            "lora": lora_text["by_column"],
            "mixed": mixed_text["by_column"],
        },
        "arrow_refhas_by_column": {
            "base": base_arrow_bycol,
            "lora": lora_arrow_bycol,
            "mixed": mixed_arrow_bycol,
        },
        "column_model_map": col_map,
    }

    compare_path = os.path.join(CFG["output_dir"], "eval_test_compare.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(out_compare, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {compare_path}")

    map_path = os.path.join(CFG["output_dir"], "column_model_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(col_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {map_path}")

    # plots: CER by column (base vs lora)
    col_order = ["cut", "time", "picture", "action_memo", "dialogue"]
    col_order = [c for c in col_order if c in CFG["eval_columns"]]

    base_cer_vals = [float(base_text["by_column"].get(c, {}).get("cer", 0.0)) for c in col_order]
    lora_cer_vals = [float(lora_text["by_column"].get(c, {}).get("cer", 0.0)) for c in col_order]
    plot_bar_base_vs_lora(
        col_order, base_cer_vals, lora_cer_vals,
        ylabel="CER (lower is better)",
        title="Text CER by column (base vs lora) on test",
        out_path=os.path.join(CFG["output_dir"], "bar_text_cer.png"),
    )

    # plots: arrow seq edit rate (ref_has only) by column
    arrow_cols = ["picture", "action_memo", "dialogue"]
    base_arrow_edit = [float(base_arrow_bycol.get(c, {}).get("arrow_seq_edit_rate_micro", 0.0)) for c in arrow_cols]
    lora_arrow_edit = [float(lora_arrow_bycol.get(c, {}).get("arrow_seq_edit_rate_micro", 0.0)) for c in arrow_cols]
    plot_bar_base_vs_lora(
        arrow_cols, base_arrow_edit, lora_arrow_edit,
        ylabel="Arrow Seq Edit Rate (lower is better)",
        title="Arrow seq edit rate (ref_has only) by column (base vs lora)",
        out_path=os.path.join(CFG["output_dir"], "bar_arrow_seq_edit_rate_refhas.png"),
    )

    print("[DONE] base vs lora vs mixed evaluation completed.")
    print("column_model_map =", col_map)


    # Save log history
    log_path = os.path.join(CFG["output_dir"], CFG["log_history_filename"])
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote log_history: {log_path}")

    # Plot loss curve
    loss_path = os.path.join(CFG["output_dir"], CFG["loss_plot_filename"])
    plot_loss_curve(trainer.state.log_history, loss_path)


if __name__ == "__main__":
    main()

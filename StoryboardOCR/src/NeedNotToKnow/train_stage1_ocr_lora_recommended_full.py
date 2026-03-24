# train_stage1_ocr_lora_recommended_full.py
# ------------------------------------------------------------
# Stage1 OCR LoRA training (Qwen3-VL) focused on handwriting/arrows.
#
# ✅ Implements:
# - Split train/val/test by CUT (no leakage)
# - Column-specific OCR prompts (cut/picture/action_memo/dialogue/time)
# - LoRA target closer to multimodal connector (vision projector / merger / projector)
# - Data augmentation for handwriting (contrast/binarize/rotate/blur/noise)
# - Early stopping by validation CER (generation-based, with robust batch sanitization)
# - Base vs LoRA comparison on SAME test set
# - Column-wise bar charts (base vs lora)
# - Arrow-specific metrics (subset where ref contains arrows)
# - Top-N improved / worsened examples export (JSONL)
# - Optional hybrid inference: use BASE for chosen columns (e.g., cut/time)
#
# Notes:
# - Requires: torch, transformers, peft, pillow, tqdm, matplotlib
# - torchvision is OPTIONAL (we only use PIL-based augmentation).
# ------------------------------------------------------------

import os
import csv
import json
import math
import random
import shutil
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer, TrainerCallback

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # ---------------------
    # Models
    # ---------------------
    "base_model_id": "Qwen/Qwen3-VL-4B-Instruct",
    # "base_model_id": "Qwen/Qwen3-VL-8B-Instruct",

    # ---------------------
    # Data
    # ---------------------
    "csv_path": "data/episode01/annotation_table.csv",
    "episode_dir": "data/episode01",

    # Which columns to train/eval
    # Recommended: focus on handwriting-heavy columns first
    "train_columns": {"action_memo", "dialogue", "picture"},
    # If you insist, you can add {"cut", "time"} but it often degrades.
    # "train_columns": {"action_memo", "dialogue", "picture", "cut", "time"},

    # Evaluation columns for reporting
    "eval_columns": {"cut", "time", "action_memo", "dialogue", "picture"},

    # Hybrid inference after FT:
    # Use BASE model for these columns when evaluating "lora_hybrid"
    # (This matches your idea: keep cut/time base.)
    "use_base_for_columns_after_ft": {"cut", "time"},

    # Filters
    "skip_empty_gt": True,
    "skip_placeholder_only": True,
    "placeholders": {"□", "△"},  # treated as "unreadable but neutral" in metrics
    "neutral_symbols": ["□", "△"],
    "space_insensitive_metrics": True,  # collapse whitespace in metrics

    # Selection by cut/page (for BOTH training and evaluation rows selection)
    "select_cuts": True,
    "cut_start": 1,
    "cut_end": 130,

    "select_pages": False,
    "page_start": 1,
    "page_end": 9999,

    # ---------------------
    # Split (by CUT)
    # ---------------------
    "seed": 32,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # ---------------------
    # Training
    # ---------------------
    "output_dir": "outputs_ft/stage1_ocr_lora_recommended",
    "num_train_epochs": 10,                  # upper bound; early stopping will likely stop earlier
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,

    # checkpoint/log
    "logging_steps": 1,
    "save_steps": 10,
    "save_total_limit": 2,

    # precision / memory
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "torch_compile": False,

    # ---------------------
    # LoRA (focus on multimodal connector)
    # ---------------------
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,

        # "vision_projector" tries to target linear layers whose FULL NAME includes these keywords
        # This is intended to bias LoRA toward the multimodal connector path.
        "target_mode": "vision_projector",  # "vision_projector" | "language_all"
        "vision_name_keywords": ["projector", "merger", "mm_projector", "multi_modal", "vision_proj", "vision_projector"],
        # fallback language targets if vision targeting fails
        "fallback_language_leaf_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },

    # ---------------------
    # Sampling weights (optional but recommended)
    # ---------------------
    "use_weighted_sampling": True,
    "column_weights": {
        "action_memo": 4.0,
        "dialogue": 4.0,
        "picture": 3.0,
        "cut": 1.0,
        "time": 1.0,
    },
    "arrow_bonus_weight": 1.25,  # extra weight if GT contains arrows

    # ---------------------
    # Augmentation (handwriting)
    # ---------------------
    "augment": {
        "enable": True,
        # apply stronger aug more on handwriting-heavy columns
        "handwriting_cols": {"action_memo", "dialogue", "picture"},

        # Mix original + enhanced views by duplicating each sample
        "duplicate_views": True,     # doubles training set entries (recommended if data small)
        "enhanced_view_prob": 1.0,   # if duplicate_views, second view always enhanced

        # per-sample random augment knobs
        "p_autocontrast": 0.4,
        "p_contrast": 0.5,
        "contrast_range": (1.2, 2.2),

        "p_binarize": 0.35,
        "binarize_thresh_range": (140, 210),

        "p_rotate": 0.35,
        "max_rotate_deg": 2.5,

        "p_blur": 0.2,
        "blur_radius_range": (0.0, 1.2),

        "p_noise": 0.25,
        "noise_sigma_range": (2.0, 8.0),  # pixel-space noise (0-255 scale)
    },

    # ---------------------
    # Prompts (column-specific)
    # ---------------------
    "prompts": {
        # You can rewrite these later freely.
        # Keep them strict: "Return ONLY the transcribed text."
        "cut": None,
        "time": None,
        "picture": None,
        "action_memo": None,
        "dialogue": None,
    },

    # generation (for evaluation)
    "gen": {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    },
    "max_new_tokens_by_column": {
        "cut": 16,
        "time": 16,
        "picture": 64,
        "action_memo": 128,
        "dialogue": 128,
    },

    # early stopping (by validation CER)
    "early_stop": {
        "enable": True,
        "patience_evals": 2,      # stop if no CER improvement for N evals
        "min_delta": 1e-4,        # CER must improve by at least this
        "eval_every_steps": 10000,  # run CER eval every N train steps (set <= save_steps is convenient)
        "max_eval_samples": 100,  # cap val samples for speed
        "save_best_dirname": "best_adapter",
    },

    # test compare after training
    "run_test_compare": True,
    "max_test_samples": None,  # None => all

    # output examples
    "top_n_examples": 50,
    "copy_example_images": False,

    # debug
    "print_lora_targets": True,
}
# =========================


# -------------------------
# Basic helpers
# -------------------------
ARROWS = ["→", "←", "↑", "↓"]
ARROW_RE = re.compile(r"[→←↑↓]")

REQUIRED_COLUMNS = {"cut", "page", "row", "column", "final_gt"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV is empty.")
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return rows


def reconstruct_image_path(episode_dir: str, column: str, page: str, row: str) -> str:
    return os.path.join(episode_dir, column, f"page{page}_row{row}.png")


def safe_int(x: str) -> Optional[int]:
    try:
        return int(str(x))
    except Exception:
        return None


def in_cut_range(cut_i: int) -> bool:
    if not CFG["select_cuts"]:
        return True
    return CFG["cut_start"] <= cut_i <= CFG["cut_end"]


def in_page_range(page_i: int) -> bool:
    if not CFG["select_pages"]:
        return True
    return CFG["page_start"] <= page_i <= CFG["page_end"]


def is_placeholder_only(text: str, placeholders: Set[str]) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    t2 = "".join(ch for ch in t if not ch.isspace())
    if not t2:
        return True
    return all(ch in placeholders for ch in t2)


# -------------------------
# Prompts (column-specific)
# -------------------------
def build_prompts_if_needed():
    # If user didn't fill CFG["prompts"][col], we set recommended strict defaults.
    if CFG["prompts"]["cut"] is None:
        CFG["prompts"]["cut"] = (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: cut.\n"
            "This region contains a cut number and sometimes a time.\n"
            "Handwritten digits may appear.\n\n"
            "Task: Transcribe ALL visible text as faithfully as possible.\n"
            "Rules:\n"
            "- Keep original line breaks.\n"
            "- Do NOT correct typos.\n"
            "- Do NOT translate.\n"
            "- If unreadable, output '□'.\n"
            "- Return ONLY the transcribed text. No explanations.\n"
        )

    if CFG["prompts"]["time"] is None:
        CFG["prompts"]["time"] = (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: time.\n"
            "This region mainly contains a time code like 05:00.\n"
            "Handwritten digits may appear.\n\n"
            "Task: Transcribe ALL visible text as faithfully as possible.\n"
            "Rules:\n"
            "- Keep original line breaks.\n"
            "- Preserve ':' exactly.\n"
            "- Do NOT correct typos.\n"
            "- Do NOT translate.\n"
            "- If unreadable, output '□'.\n"
            "- Return ONLY the transcribed text. No explanations.\n"
        )

    if CFG["prompts"]["picture"] is None:
        CFG["prompts"]["picture"] = (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: picture.\n"
            "This region is mostly drawings, but may include handwritten labels.\n"
            "Handwritten labels often include arrows and short production terms.\n\n"
            "Task: Transcribe ALL visible text as faithfully as possible.\n"
            "Rules:\n"
            "- Keep original line breaks.\n"
            "- Preserve symbols exactly (→ ← ↑ ↓, punctuation).\n"
            "- Do NOT correct typos.\n"
            "- Do NOT translate.\n"
            "- If unreadable, output '□'.\n"
            "- Return ONLY the transcribed text. No explanations.\n"
        )

    if CFG["prompts"]["action_memo"] is None:
        CFG["prompts"]["action_memo"] = (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: action_memo.\n"
            "This region contains handwritten notes and many production terms.\n"
            "Handwriting may include arrows and symbols.\n\n"
            "Task: Transcribe ALL visible text as faithfully as possible.\n"
            "Rules:\n"
            "- Keep original line breaks.\n"
            "- Preserve symbols exactly (→ ← ↑ ↓, punctuation).\n"
            "- Do NOT correct typos.\n"
            "- Do NOT translate.\n"
            "- If unreadable, output '□'.\n"
            "- Return ONLY the transcribed text. No explanations.\n"
        )

    if CFG["prompts"]["dialogue"] is None:
        CFG["prompts"]["dialogue"] = (
            "You are an OCR engine for anime storyboards.\n"
            "Target column: dialogue.\n"
            "This region contains dialogue lines, speaker names, and sometimes production terms.\n"
            "Handwriting may include arrows and symbols.\n\n"
            "Task: Transcribe ALL visible text as faithfully as possible.\n"
            "Rules:\n"
            "- Keep original line breaks.\n"
            "- Preserve symbols exactly (→ ← ↑ ↓, punctuation).\n"
            "- Do NOT correct typos.\n"
            "- Do NOT translate.\n"
            "- If unreadable, output '□'.\n"
            "- Return ONLY the transcribed text. No explanations.\n"
        )


def prompt_for_column(col: str) -> str:
    build_prompts_if_needed()
    return CFG["prompts"].get(col) or CFG["prompts"]["dialogue"]


def max_new_tokens_for_column(col: str) -> int:
    return int(CFG["max_new_tokens_by_column"].get(col, 128))


# -------------------------
# Augmentation (PIL-based)
# -------------------------
def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    # sigma in pixel scale (0..255)
    if sigma <= 0:
        return img
    x = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(img.size[1], img.size[0], 3)
         .numpy())
    ).float()
    noise = torch.randn_like(x) * float(sigma)
    y = torch.clamp(x + noise, 0, 255).byte().numpy()
    return Image.fromarray(y, mode="RGB")


def apply_handwriting_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    aug = CFG["augment"]
    out = img

    # autocontrast
    if rng.random() < aug["p_autocontrast"]:
        out = ImageOps.autocontrast(out)

    # contrast
    if rng.random() < aug["p_contrast"]:
        cmin, cmax = aug["contrast_range"]
        factor = rng.uniform(cmin, cmax)
        out = ImageEnhance.Contrast(out).enhance(factor)

    # binarize-like
    if rng.random() < aug["p_binarize"]:
        th_min, th_max = aug["binarize_thresh_range"]
        thr = rng.randint(int(th_min), int(th_max))
        g = out.convert("L")
        # threshold -> 0/255, then back to RGB
        bw = g.point(lambda p: 255 if p >= thr else 0)
        out = bw.convert("RGB")

    # small rotation
    if rng.random() < aug["p_rotate"]:
        deg = rng.uniform(-aug["max_rotate_deg"], aug["max_rotate_deg"])
        out = out.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    # blur
    if rng.random() < aug["p_blur"]:
        rmin, rmax = aug["blur_radius_range"]
        radius = rng.uniform(rmin, rmax)
        if radius > 0:
            out = out.filter(ImageFilter.GaussianBlur(radius=radius))

    # noise
    if rng.random() < aug["p_noise"]:
        smin, smax = aug["noise_sigma_range"]
        sigma = rng.uniform(smin, smax)
        out = add_gaussian_noise(out, sigma=sigma)

    return out


# -------------------------
# Dataset (LoRA training)
# -------------------------
class OCRTrainDataset(Dataset):
    """
    Each sample is a single crop image (column/page/row) with a column-specific prompt and GT text.
    Optionally duplicates samples to mix original+enhanced views.
    """
    def __init__(self, rows: List[Dict[str, str]], processor: AutoProcessor, is_train: bool):
        self.processor = processor
        self.is_train = is_train

        # Expand rows to include duplicate views if enabled
        self.items: List[Tuple[Dict[str, str], str]] = []
        dup = CFG["augment"]["enable"] and is_train and CFG["augment"]["duplicate_views"]
        for r in rows:
            self.items.append((r, "orig"))
            if dup:
                self.items.append((r, "enh"))

        # RNG per dataset
        self.rng = random.Random(CFG["seed"] + (0 if is_train else 999))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r, view = self.items[idx]
        col = r["column"]
        page = r["page"]
        row = r["row"]
        gt = (r.get("final_gt") or "")

        img_path = reconstruct_image_path(CFG["episode_dir"], col, page, row)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        # augmentation only for train + handwriting columns
        if self.is_train and CFG["augment"]["enable"] and (col in CFG["augment"]["handwriting_cols"]):
            if view == "enh":
                # always enhanced view (recommended)
                img = apply_handwriting_augment(img, self.rng)
            else:
                # orig view can still get light random aug if you want; keep conservative
                # (set probabilities low by editing apply_handwriting_augment or wrap here)
                pass

        prompt = prompt_for_column(col)

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

        # IMPORTANT: keep only model-acceptable tensor keys
        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # vision tensors (usually pixel_values, image_grid_thw)
        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = v[0]

        return out


@dataclass
class DataCollatorForVLStrict:
    """
    Strict collator:
    - pads input_ids/attention_mask/labels
    - stacks known vision tensors
    - filters to allowed keys to avoid forward() surprises
    """
    processor: AutoProcessor
    debug_first_n: int = 0
    _calls: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        self._calls += 1
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
        padded_labels = torch.full(
            (len(labels), max_len),
            fill_value=-100,
            dtype=batch["input_ids"].dtype
        )
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        batch["labels"] = padded_labels

        # Collect vision keys present in features (e.g., pixel_values, image_grid_thw)
        # Stack only tensor values with same shape per sample.
        for k in list(features[0].keys()):
            if k in ("input_ids", "attention_mask", "labels"):
                continue
            vals = [f[k] for f in features if k in f]
            if not vals:
                continue
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals, dim=0)

        # Filter to known keys (safer than passing unknown stuff)
        allowed = {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
        batch = {k: v for k, v in batch.items() if k in allowed}

        if self.debug_first_n and self._calls <= self.debug_first_n:
            print("\n" + "=" * 80)
            print(f"[CollatorDebug] call={self._calls}  batch_size={len(features)}")
            print("[CollatorDebug] batch keys:", sorted(batch.keys()))
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"   {k} -> shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            print("=" * 80)

        return batch


# -------------------------
# Weighted sampling
# -------------------------
def sample_weights_from_rows(rows: List[Dict[str, str]]) -> List[float]:
    weights: List[float] = []
    for r in rows:
        col = r.get("column", "")
        w = float(CFG["column_weights"].get(col, 1.0))
        gt = (r.get("final_gt") or "")
        if any(a in gt for a in ARROWS):
            w *= float(CFG["arrow_bonus_weight"])
        weights.append(w)
    return weights


class WeightedTrainer(Trainer):
    def __init__(self, *args, train_rows_for_weights: Optional[List[Dict[str, str]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_rows_for_weights = train_rows_for_weights

    def get_train_dataloader(self) -> DataLoader:
        if not CFG["use_weighted_sampling"]:
            return super().get_train_dataloader()

        if self._train_rows_for_weights is None:
            return super().get_train_dataloader()

        weights = sample_weights_from_rows(self._train_rows_for_weights)
        # If dataset duplicates views, weights must match dataset length:
        # We approximate by repeating weights equally (orig/enh).
        if len(weights) != len(self.train_dataset):
            if len(self.train_dataset) % len(weights) == 0:
                rep = len(self.train_dataset) // len(weights)
                weights = [w for w in weights for _ in range(rep)]
            else:
                # fallback (no weighting)
                return super().get_train_dataloader()

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )


# -------------------------
# Split by CUT (no leakage)
# -------------------------
def split_by_cut(rows: List[Dict[str, str]], seed: int, val_ratio: float, test_ratio: float) -> Tuple[List, List, List, Dict[str, Any]]:
    cut_ids = sorted({r["cut"] for r in rows if r.get("cut")})
    rnd = random.Random(seed)
    rnd.shuffle(cut_ids)

    n = len(cut_ids)
    if n == 0:
        return [], [], [], {"train_cut_ids": [], "val_cut_ids": [], "test_cut_ids": []}

    n_val = max(1, int(n * val_ratio)) if n >= 10 else max(0, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio)) if n >= 10 else max(0, int(n * test_ratio))
    if n_val + n_test >= n:
        n_val = max(0, n // 10)
        n_test = max(0, n // 10)

    val_set = set(cut_ids[:n_val])
    test_set = set(cut_ids[n_val:n_val + n_test])
    train_set = set(cut_ids[n_val + n_test:])

    train_rows, val_rows, test_rows = [], [], []
    for r in rows:
        c = r["cut"]
        if c in val_set:
            val_rows.append(r)
        elif c in test_set:
            test_rows.append(r)
        else:
            train_rows.append(r)

    info = {
        "train_cut_ids": sorted(train_set),
        "val_cut_ids": sorted(val_set),
        "test_cut_ids": sorted(test_set),
        "n_cuts": n,
    }
    return train_rows, val_rows, test_rows, info


# -------------------------
# Metrics (space-insensitive, neutral symbols removed)
# -------------------------
def normalize_for_metrics(s: str, neutral_symbols: List[str]) -> str:
    s = s or ""
    # convert newlines/tabs to spaces for evaluation
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for sym in neutral_symbols:
        s = s.replace(sym, "")
    if CFG["space_insensitive_metrics"]:
        s = " ".join(s.split())
    return s.strip()


def neutral_ratio(s: str, neutral_symbols: List[str]) -> float:
    s = s or ""
    denom = max(len(s), 1)
    cnt = 0
    for sym in neutral_symbols:
        cnt += s.count(sym)
    return float(cnt) / float(denom)


def levenshtein(a: List[str], b: List[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer_raw(hyp: str, ref: str, neutral_symbols: List[str]) -> float:
    h = list(normalize_for_metrics(hyp, neutral_symbols))
    r = list(normalize_for_metrics(ref, neutral_symbols))
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    # NOTE: this can exceed 1.0 if hyp is much longer than ref.
    return float(levenshtein(h, r)) / float(len(r))


def cer_clipped(hyp: str, ref: str, neutral_symbols: List[str]) -> float:
    return min(1.0, max(0.0, cer_raw(hyp, ref, neutral_symbols)))


def exact_match(hyp: str, ref: str, neutral_symbols: List[str]) -> int:
    return int(normalize_for_metrics(hyp, neutral_symbols) == normalize_for_metrics(ref, neutral_symbols))


def char_accuracy(hyp: str, ref: str, neutral_symbols: List[str]) -> float:
    # bounded 0..1 view
    return 1.0 - cer_clipped(hyp, ref, neutral_symbols)


def arrows_only(s: str) -> str:
    return "".join(ARROW_RE.findall(s or ""))


# -------------------------
# Robust generation helper (Qwen3-VL)
# -------------------------
def sanitize_batch_for_generate(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v

    # squeeze accidental (1,1,L) -> (1,L) for ids/masks
    for k in ("input_ids", "attention_mask"):
        if k in out and isinstance(out[k], torch.Tensor):
            x = out[k]
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out[k] = x

    # vision tensors sometimes come with (1,1,...) too
    for k in ("pixel_values", "image_grid_thw"):
        if k in out and isinstance(out[k], torch.Tensor):
            x = out[k]
            if x.dim() >= 3 and x.size(1) == 1:
                x = x.squeeze(1)
            out[k] = x

    return out


@torch.no_grad()
def generate_one_qwen3vl(model, processor, img: Image.Image, col: str) -> str:
    prompt = prompt_for_column(col)

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

    batch = sanitize_batch_for_generate(batch, model.device)

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    gen = dict(CFG["gen"])
    gen["max_new_tokens"] = max_new_tokens_for_column(col)

    # primary attempt
    try:
        out_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            **{k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")},
            **gen,
        )[0]
        prompt_len = batch["input_ids"].shape[1]
        return tok.decode(out_ids[prompt_len:], skip_special_tokens=True).strip()

    except Exception as e1:
        # fallback attempt: try squeezing batch dim (some envs behave better)
        try:
            ids = batch["input_ids"]
            msk = batch.get("attention_mask", None)
            if isinstance(ids, torch.Tensor) and ids.dim() == 2 and ids.size(0) == 1:
                ids2 = ids[0]
                msk2 = msk[0] if (isinstance(msk, torch.Tensor) and msk.dim() == 2 and msk.size(0) == 1) else msk
                extra = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}
                out_ids = model.generate(
                    input_ids=ids2,
                    attention_mask=msk2,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    **extra,
                    **gen,
                )[0]
                # if ids2 is 1D, prompt_len is its length
                prompt_len = int(ids2.shape[0]) if ids2.dim() == 1 else int(ids2.shape[1])
                return tok.decode(out_ids[prompt_len:], skip_special_tokens=True).strip()
        except Exception:
            pass

        # last resort: return empty (so metrics reflect failure but training doesn't crash)
        return ""


# -------------------------
# Aggregation (micro + macro by cut + by column + arrow subset)
# -------------------------
def agg_init() -> Dict[str, Any]:
    return {
        "n": 0,
        "sum_edit": 0.0,
        "sum_ref_len": 0.0,
        "sum_exact": 0.0,
        "sum_char_acc": 0.0,
        "sum_neutral_h": 0.0,
        "sum_neutral_r": 0.0,
    }


def update_agg(agg: Dict[str, Any], hyp: str, ref: str, neutral_symbols: List[str]):
    hyp_n = normalize_for_metrics(hyp, neutral_symbols)
    ref_n = normalize_for_metrics(ref, neutral_symbols)

    h = list(hyp_n)
    r = list(ref_n)

    # edit distance + ref len for true micro CER
    ed = float(levenshtein(h, r))
    rl = float(len(r))

    agg["n"] += 1
    agg["sum_edit"] += ed
    agg["sum_ref_len"] += rl
    agg["sum_exact"] += float(exact_match(hyp, ref, neutral_symbols))
    agg["sum_char_acc"] += float(char_accuracy(hyp, ref, neutral_symbols))
    agg["sum_neutral_h"] += float(neutral_ratio(hyp, neutral_symbols))
    agg["sum_neutral_r"] += float(neutral_ratio(ref, neutral_symbols))


def finalize_agg(agg: Dict[str, Any]) -> Dict[str, float]:
    n = max(int(agg["n"]), 1)
    ref_len = float(agg["sum_ref_len"])
    cer_micro = float(agg["sum_edit"] / ref_len) if ref_len > 0 else 0.0
    return {
        "n": int(agg["n"]),
        "cer": cer_micro,
        "exact_match": float(agg["sum_exact"] / n),
        "char_accuracy": float(agg["sum_char_acc"] / n),
        "neutral_ratio_hyp_mean": float(agg["sum_neutral_h"] / n),
        "neutral_ratio_ref_mean": float(agg["sum_neutral_r"] / n),
    }


def evaluate_model_on_rows(
    *,
    model,
    processor,
    rows: List[Dict[str, str]],
    name: str,
    base_model_for_hybrid=None,
    base_for_columns: Optional[Set[str]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    neutral = CFG["neutral_symbols"]
    base_for_columns = base_for_columns or set()

    total = agg_init()
    by_col: Dict[str, Any] = {}
    by_cut_micro: Dict[str, Any] = {}
    arrow_subset = agg_init()

    samples_out: List[Dict[str, Any]] = []

    rows2 = rows[:max_samples] if (max_samples is not None) else rows

    for idx, r in enumerate(tqdm(rows2, desc=f"Eval {name}", leave=False)):
        col = r["column"]
        ref = r["final_gt"]
        img_path = reconstruct_image_path(CFG["episode_dir"], col, r["page"], r["row"])
        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path).convert("RGB")

        # choose model (hybrid option)
        use_model = model
        use_proc = processor
        if (col in base_for_columns) and (base_model_for_hybrid is not None):
            use_model = base_model_for_hybrid
            # processor should be compatible (same base), so keep processor

        hyp = generate_one_qwen3vl(use_model, use_proc, img, col)

        # per-sample metrics
        cer_r = cer_raw(hyp, ref, neutral)
        cer_c = cer_clipped(hyp, ref, neutral)
        em = exact_match(hyp, ref, neutral)
        ca = char_accuracy(hyp, ref, neutral)
        nr = neutral_ratio(ref, neutral)
        nh = neutral_ratio(hyp, neutral)

        update_agg(total, hyp, ref, neutral)

        if col not in by_col:
            by_col[col] = agg_init()
        update_agg(by_col[col], hyp, ref, neutral)

        cut = r.get("cut", "")
        if cut not in by_cut_micro:
            by_cut_micro[cut] = agg_init()
        update_agg(by_cut_micro[cut], hyp, ref, neutral)

        # arrow subset
        if any(a in (ref or "") for a in ARROWS):
            update_agg(arrow_subset, hyp, ref, neutral)

        samples_out.append({
            "id": idx,
            "cut": r.get("cut"),
            "page": r.get("page"),
            "row": r.get("row"),
            "column": col,
            "image_path": img_path,
            "ref": ref,
            "hyp": hyp,
            "cer_raw": float(cer_r),
            "cer_clipped": float(cer_c),
            "exact_match": int(em),
            "char_accuracy": float(ca),
            "neutral_ratio_ref": float(nr),
            "neutral_ratio_hyp": float(nh),
            "arrows_ref": arrows_only(ref),
            "arrows_hyp": arrows_only(hyp),
            "arrow_only_exact": int(arrows_only(ref) == arrows_only(hyp)),
        })

    total_f = finalize_agg(total)
    by_col_f = {k: finalize_agg(v) for k, v in by_col.items()}

    # macro by cut: average per-cut CER/Exact/CharAcc over cuts
    cut_metrics = {k: finalize_agg(v) for k, v in by_cut_micro.items()}
    if cut_metrics:
        macro_cer = sum(m["cer"] for m in cut_metrics.values()) / max(len(cut_metrics), 1)
        macro_em = sum(m["exact_match"] for m in cut_metrics.values()) / max(len(cut_metrics), 1)
        macro_ca = sum(m["char_accuracy"] for m in cut_metrics.values()) / max(len(cut_metrics), 1)
    else:
        macro_cer, macro_em, macro_ca = 0.0, 0.0, 0.0

    arrow_f = finalize_agg(arrow_subset)
    # arrow-only strict metrics
    arrow_n = 0
    arrow_only_exact_sum = 0
    for s in samples_out:
        if (s.get("arrows_ref") or "") != "":
            arrow_n += 1
            arrow_only_exact_sum += int(s.get("arrow_only_exact", 0))
    arrow_only_exact = float(arrow_only_exact_sum / max(arrow_n, 1))

    report = {
        "name": name,
        "total_micro": total_f,
        "macro_by_cut": {
            "n_cuts": len(cut_metrics),
            "macro_cer": float(macro_cer),
            "macro_exact_match": float(macro_em),
            "macro_char_accuracy": float(macro_ca),
        },
        "by_column_micro": by_col_f,
        "arrow_subset_micro": arrow_f,
        "arrow_only_exact_match": float(arrow_only_exact),
        "meta": {
            "created_at": now_iso(),
            "space_insensitive_metrics": CFG["space_insensitive_metrics"],
            "neutral_symbols": CFG["neutral_symbols"],
        }
    }
    return report, samples_out


# -------------------------
# Plotting (base vs lora bar charts)
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


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def maybe_copy_image(src_path: str, dst_dir: str, dst_name: str):
    if not CFG["copy_example_images"]:
        return
    ensure_dir(dst_dir)
    if os.path.isfile(src_path):
        shutil.copy2(src_path, os.path.join(dst_dir, dst_name))


# -------------------------
# LoRA target selection (bias toward multimodal connector)
# -------------------------
def find_linear_module_names(model) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names


def pick_lora_targets(model) -> List[str]:
    mode = CFG["lora"]["target_mode"]
    if mode == "language_all":
        return list(CFG["lora"]["fallback_language_leaf_targets"])

    # vision_projector mode: pick FULL module names that include keywords
    kws = [k.lower() for k in CFG["lora"]["vision_name_keywords"]]
    full_linear_names = find_linear_module_names(model)

    selected_full = []
    for n in full_linear_names:
        nl = n.lower()
        if any(k in nl for k in kws):
            selected_full.append(n)

    # If nothing found, fallback
    if not selected_full:
        return list(CFG["lora"]["fallback_language_leaf_targets"])

    return sorted(selected_full)


def wrap_with_lora(base_model):
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA training but not installed.")

    targets = pick_lora_targets(base_model)

    lora_cfg = LoraConfig(
        r=int(CFG["lora"]["r"]),
        lora_alpha=int(CFG["lora"]["alpha"]),
        lora_dropout=float(CFG["lora"]["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    try:
        model = get_peft_model(base_model, lora_cfg)
        if CFG["print_lora_targets"]:
            print("[LoRA] target_modules (as given) =", targets[:50], ("..." if len(targets) > 50 else ""))
        return model, targets
    except Exception as e:
        # fallback to leaf names (last component) to maximize compatibility
        leaf = sorted({t.split(".")[-1] for t in targets})
        print("[LoRA] FAILED with full names. Fallback to leaf names:", leaf)
        lora_cfg = LoraConfig(
            r=int(CFG["lora"]["r"]),
            lora_alpha=int(CFG["lora"]["alpha"]),
            lora_dropout=float(CFG["lora"]["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=leaf,
        )
        model = get_peft_model(base_model, lora_cfg)
        return model, leaf


# -------------------------
# Early stopping callback (validation CER)
# -------------------------
class CEREarlyStopCallback(TrainerCallback):
    def __init__(self, processor, val_rows: List[Dict[str, str]], base_model_for_hybrid=None):
        self.processor = processor
        self.val_rows = val_rows
        self.base_model_for_hybrid = base_model_for_hybrid

        self.best_cer = float("inf")
        self.bad_count = 0
        self.best_dir = os.path.join(CFG["output_dir"], CFG["early_stop"]["save_best_dirname"])

    def on_step_end(self, args, state, control, **kwargs):
        if not CFG["early_stop"]["enable"]:
            return control
        step = int(state.global_step)
        every = int(CFG["early_stop"]["eval_every_steps"])
        if every <= 0:
            return control
        if step == 0 or (step % every != 0):
            return control
        if not self.val_rows:
            return control

        trainer: Trainer = kwargs["trainer"]
        model = trainer.model

        # Temporarily switch to eval
        was_train = model.training
        model.eval()

        # Evaluate CER on val (use LoRA model; do not hybridize by default)
        max_samples = int(CFG["early_stop"]["max_eval_samples"]) if CFG["early_stop"]["max_eval_samples"] else None
        report, _ = evaluate_model_on_rows(
            model=model,
            processor=self.processor,
            rows=self.val_rows,
            name="val_for_early_stop",
            base_model_for_hybrid=None,
            base_for_columns=set(),  # early stop should reflect LoRA effect on trained columns
            max_samples=max_samples,
        )
        cur_cer = float(report["total_micro"]["cer"])
        trainer.log({"val_gen_cer": cur_cer})

        # Early stopping logic (lower CER is better)
        min_delta = float(CFG["early_stop"]["min_delta"])
        patience = int(CFG["early_stop"]["patience_evals"])

        improved = (self.best_cer - cur_cer) > min_delta
        if improved:
            self.best_cer = cur_cer
            self.bad_count = 0
            # save best adapter
            ensure_dir(self.best_dir)
            trainer.save_model(self.best_dir)
            print(f"[EarlyStop] step={step}  NEW BEST val CER={cur_cer:.6f}  saved -> {self.best_dir}")
        else:
            self.bad_count += 1
            print(f"[EarlyStop] step={step}  val CER={cur_cer:.6f}  (best={self.best_cer:.6f})  bad_count={self.bad_count}/{patience}")
            if self.bad_count >= patience:
                print("[EarlyStop] stopping training (no improvement).")
                control.should_training_stop = True

        if was_train:
            model.train()
        return control


# -------------------------
# Loss plot (train / eval_loss if available)
# -------------------------
def plot_loss_curve(log_history: List[Dict[str, Any]], out_path: str):
    steps_train, loss_train = [], []
    steps_eval, loss_eval = [], []

    for item in log_history:
        if "loss" in item and "step" in item and "eval_loss" not in item:
            steps_train.append(item["step"])
            loss_train.append(item["loss"])
        if "eval_loss" in item and "step" in item:
            steps_eval.append(item["step"])
            loss_eval.append(item["eval_loss"])

    if not steps_train and not steps_eval:
        return

    plt.figure()
    if steps_train:
        plt.plot(steps_train, loss_train, label="train_loss")
    if steps_eval:
        plt.plot(steps_eval, loss_eval, label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    set_seed(int(CFG["seed"]))
    ensure_dir(CFG["output_dir"])

    # 1) load + filter rows
    rows_all = read_csv_rows(CFG["csv_path"])

    rows: List[Dict[str, str]] = []
    for r in rows_all:
        col = r.get("column", "")
        if col not in CFG["eval_columns"]:
            continue

        gt = (r.get("final_gt") or "")
        if CFG["skip_empty_gt"] and not gt.strip():
            continue
        if CFG["skip_placeholder_only"] and is_placeholder_only(gt, CFG["placeholders"]):
            continue

        cut_i = safe_int(r.get("cut", ""))
        page_i = safe_int(r.get("page", ""))
        row_i = safe_int(r.get("row", ""))

        if cut_i is None or page_i is None or row_i is None:
            continue
        if not in_cut_range(cut_i):
            continue
        if not in_page_range(page_i):
            continue

        # must have corresponding crop image
        img_path = reconstruct_image_path(CFG["episode_dir"], col, r["page"], r["row"])
        if not os.path.isfile(img_path):
            continue

        rows.append(r)

    if not rows:
        raise RuntimeError("No usable rows after filtering. Check ranges, CSV, and crop images.")

    # 2) split by cut (no leakage)
    train_rows_allcols, val_rows_allcols, test_rows_allcols, split_info = split_by_cut(
        rows=rows,
        seed=int(CFG["seed"]),
        val_ratio=float(CFG["val_ratio"]),
        test_ratio=float(CFG["test_ratio"]),
    )

    # Training uses subset of columns
    train_rows = [r for r in train_rows_allcols if r["column"] in CFG["train_columns"]]
    val_rows = [r for r in val_rows_allcols if r["column"] in CFG["train_columns"]]  # early stop focuses on trained cols
    test_rows = [r for r in test_rows_allcols if r["column"] in CFG["eval_columns"]]

    if not train_rows:
        raise RuntimeError("Train split is empty after filtering to train_columns. Add more labeled cuts or widen range.")

    # save split for reproducibility
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
                "skip_placeholder_only": CFG["skip_placeholder_only"],
            },
            "counts": {
                "train_rows_allcols": len(train_rows_allcols),
                "val_rows_allcols": len(val_rows_allcols),
                "test_rows_allcols": len(test_rows_allcols),
                "train_rows_traincols": len(train_rows),
                "val_rows_traincols": len(val_rows),
                "test_rows_evalcols": len(test_rows),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {split_path}")

    # 3) load base model + processor
    processor = AutoProcessor.from_pretrained(CFG["base_model_id"])
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if CFG["bf16"] else "auto",
    )

    if CFG["gradient_checkpointing"] and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        # training stability: disable cache
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    if CFG["torch_compile"] and hasattr(torch, "compile"):
        base_model = torch.compile(base_model)

    # 4) pre-eval BASE on test (optional but recommended for honest comparison)
    ensure_dir(os.path.join(CFG["output_dir"], "eval_compare"))
    base_test_report, base_test_samples = evaluate_model_on_rows(
        model=base_model,
        processor=processor,
        rows=test_rows,
        name="base_test",
        max_samples=CFG["max_test_samples"],
    )
    base_report_path = os.path.join(CFG["output_dir"], "eval_compare", "base_test_report.json")
    with open(base_report_path, "w", encoding="utf-8") as f:
        json.dump(base_test_report, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {base_report_path}")

    # 5) wrap with LoRA
    lora_model, lora_targets = wrap_with_lora(base_model)
    if hasattr(lora_model, "print_trainable_parameters"):
        lora_model.print_trainable_parameters()

    # 6) dataset + collator
    train_ds = OCRTrainDataset(train_rows, processor, is_train=True)
    # Trainer eval_loss is not used for CER early stop; keep eval_dataset minimal or None.
    val_ds = OCRTrainDataset(val_rows, processor, is_train=False) if val_rows else None

    collator = DataCollatorForVLStrict(processor=processor, debug_first_n=0)

    # 7) TrainingArguments
    args = TrainingArguments(
        output_dir=CFG["output_dir"],
        num_train_epochs=float(CFG["num_train_epochs"]),
        per_device_train_batch_size=int(CFG["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(CFG["gradient_accumulation_steps"]),
        learning_rate=float(CFG["learning_rate"]),
        warmup_ratio=float(CFG["warmup_ratio"]),
        weight_decay=float(CFG["weight_decay"]),
        logging_steps=int(CFG["logging_steps"]),
        save_steps=int(CFG["save_steps"]),
        save_total_limit=int(CFG["save_total_limit"]),
        bf16=bool(CFG["bf16"]),
        fp16=bool(CFG["fp16"]),
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no",  # we do CER eval in callback (generation-based)
    )

    trainer_cls = WeightedTrainer if CFG["use_weighted_sampling"] else Trainer
    trainer = trainer_cls(
        model=lora_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        train_rows_for_weights=train_rows,  # weighting reference
    )

    # 8) Early stopping by CER on val (generation-based)
    if CFG["early_stop"]["enable"] and val_rows:
        trainer.add_callback(CEREarlyStopCallback(processor=processor, val_rows=val_rows))

    # 9) train
    trainer.train()

    # Save final adapter
    trainer.save_model(CFG["output_dir"])

    # Save log history + loss plot
    log_path = os.path.join(CFG["output_dir"], "log_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    loss_plot = os.path.join(CFG["output_dir"], "loss_curve.png")
    plot_loss_curve(trainer.state.log_history, loss_plot)
    print(f"[OK] wrote {log_path}")
    print(f"[OK] wrote {loss_plot}")

    # 10) Load best adapter (if exists) for test evaluation
    best_dir = os.path.join(CFG["output_dir"], CFG["early_stop"]["save_best_dirname"])
    best_exists = os.path.isdir(best_dir) and any(os.path.isfile(os.path.join(best_dir, n)) for n in os.listdir(best_dir))
    adapter_dir_for_eval = best_dir if best_exists else CFG["output_dir"]

    if not CFG["run_test_compare"]:
        print("[DONE] training finished (test compare disabled).")
        return

    # Re-load BASE fresh (safer), then load adapter
    base_for_eval = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["base_model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if CFG["bf16"] else "auto",
    )
    base_for_eval.eval()

    if PeftModel is None:
        raise RuntimeError("peft is required to evaluate LoRA adapter (PeftModel not available).")

    lora_for_eval = PeftModel.from_pretrained(base_for_eval, adapter_dir_for_eval)
    lora_for_eval.eval()

    # 11) Evaluate LoRA on test (full LoRA)
    lora_test_report, lora_test_samples = evaluate_model_on_rows(
        model=lora_for_eval,
        processor=processor,
        rows=test_rows,
        name="lora_test",
        max_samples=CFG["max_test_samples"],
    )

    # 12) Evaluate LoRA-HYBRID (base for cut/time, etc.)
    hybrid_test_report, hybrid_test_samples = evaluate_model_on_rows(
        model=lora_for_eval,
        processor=processor,
        rows=test_rows,
        name="lora_hybrid_test",
        base_model_for_hybrid=base_for_eval,
        base_for_columns=set(CFG["use_base_for_columns_after_ft"]),
        max_samples=CFG["max_test_samples"],
    )

    # Save reports
    out_dir = os.path.join(CFG["output_dir"], "eval_compare")
    ensure_dir(out_dir)

    lora_report_path = os.path.join(out_dir, "lora_test_report.json")
    with open(lora_report_path, "w", encoding="utf-8") as f:
        json.dump(lora_test_report, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {lora_report_path}")

    hybrid_report_path = os.path.join(out_dir, "lora_hybrid_test_report.json")
    with open(hybrid_report_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_test_report, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {hybrid_report_path}")

    # 13) Compare base vs lora (and base vs hybrid) + plots
    # Stable column order
    col_order = ["cut", "time", "action_memo", "dialogue", "picture"]
    col_order = [c for c in col_order if c in CFG["eval_columns"]]

    def get_col_metric(report: Dict[str, Any], col: str, key: str) -> float:
        return float((report.get("by_column_micro", {}).get(col, {}) or {}).get(key, 0.0))

    # CER plot: base vs lora (raw micro CER)
    base_cer_vals = [get_col_metric(base_test_report, c, "cer") for c in col_order]
    lora_cer_vals = [get_col_metric(lora_test_report, c, "cer") for c in col_order]
    plot_bar_base_vs_lora(
        columns_order=col_order,
        base_vals=base_cer_vals,
        lora_vals=lora_cer_vals,
        ylabel="CER (lower is better)",
        title="CER by column (base vs lora) on SAME test set",
        out_path=os.path.join(out_dir, "bar_cer_base_vs_lora.png"),
    )

    # Exact match plot
    base_em_vals = [get_col_metric(base_test_report, c, "exact_match") for c in col_order]
    lora_em_vals = [get_col_metric(lora_test_report, c, "exact_match") for c in col_order]
    plot_bar_base_vs_lora(
        columns_order=col_order,
        base_vals=base_em_vals,
        lora_vals=lora_em_vals,
        ylabel="Exact Match (higher is better)",
        title="Exact Match by column (base vs lora) on SAME test set",
        out_path=os.path.join(out_dir, "bar_exact_match_base_vs_lora.png"),
    )
    print(f"[OK] wrote plots under {out_dir}")

    # 14) Top-N improved/worsened examples (by CER delta, clipped to avoid insane domination)
    # Build a dict from (cut,page,row,column) -> sample for base/lora/hybrid
    def key_of(s: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (str(s.get("cut")), str(s.get("page")), str(s.get("row")), str(s.get("column")))

    base_map = {key_of(s): s for s in base_test_samples}
    lora_map = {key_of(s): s for s in lora_test_samples}
    hybr_map = {key_of(s): s for s in hybrid_test_samples}

    paired = []
    for k, sb in base_map.items():
        sl = lora_map.get(k)
        sh = hybr_map.get(k)
        if sl is None or sh is None:
            continue

        # Use clipped CER for ranking stability
        delta_lora = float(sl["cer_clipped"] - sb["cer_clipped"])
        delta_hybr = float(sh["cer_clipped"] - sb["cer_clipped"])

        paired.append({
            "key": k,
            "cut": sb.get("cut"),
            "page": sb.get("page"),
            "row": sb.get("row"),
            "column": sb.get("column"),
            "image_path": sb.get("image_path"),

            "ref": sb.get("ref"),
            "hyp_base": sb.get("hyp"),
            "hyp_lora": sl.get("hyp"),
            "hyp_hybrid": sh.get("hyp"),

            "cer_base_raw": float(sb["cer_raw"]),
            "cer_lora_raw": float(sl["cer_raw"]),
            "cer_hybrid_raw": float(sh["cer_raw"]),

            "cer_base_clipped": float(sb["cer_clipped"]),
            "cer_lora_clipped": float(sl["cer_clipped"]),
            "cer_hybrid_clipped": float(sh["cer_clipped"]),

            "delta_cer_lora_minus_base_clipped": delta_lora,
            "delta_cer_hybrid_minus_base_clipped": delta_hybr,

            "exact_base": int(sb["exact_match"]),
            "exact_lora": int(sl["exact_match"]),
            "exact_hybrid": int(sh["exact_match"]),

            "arrows_ref": sb.get("arrows_ref"),
            "arrows_hyp_base": sb.get("arrows_hyp"),
            "arrows_hyp_lora": sl.get("arrows_hyp"),
            "arrows_hyp_hybrid": sh.get("arrows_hyp"),
            "arrow_only_exact_base": int(sb.get("arrow_only_exact", 0)),
            "arrow_only_exact_lora": int(sl.get("arrow_only_exact", 0)),
            "arrow_only_exact_hybrid": int(sh.get("arrow_only_exact", 0)),
        })

    paired_sorted = sorted(paired, key=lambda x: x["delta_cer_lora_minus_base_clipped"])
    top_improved = paired_sorted[: int(CFG["top_n_examples"])]
    top_worsened = list(reversed(paired_sorted[-int(CFG["top_n_examples"]):]))

    imp_path = os.path.join(out_dir, "top_improved_lora_vs_base.jsonl")
    wor_path = os.path.join(out_dir, "top_worsened_lora_vs_base.jsonl")
    write_jsonl(imp_path, top_improved)
    write_jsonl(wor_path, top_worsened)
    print(f"[OK] wrote {imp_path}")
    print(f"[OK] wrote {wor_path}")

    if CFG["copy_example_images"]:
        imp_dir = os.path.join(out_dir, "top_improved_images")
        wor_dir = os.path.join(out_dir, "top_worsened_images")
        for i, ex in enumerate(top_improved, 1):
            name = f"{i:03d}_cut{ex['cut']}_p{ex['page']}_r{ex['row']}_{ex['column']}.png"
            maybe_copy_image(ex["image_path"], imp_dir, name)
        for i, ex in enumerate(top_worsened, 1):
            name = f"{i:03d}_cut{ex['cut']}_p{ex['page']}_r{ex['row']}_{ex['column']}.png"
            maybe_copy_image(ex["image_path"], wor_dir, name)
        print(f"[OK] copied images under {out_dir}")

    # 15) Save one combined summary JSON (easy to read)
    summary = {
        "meta": {
            "created_at": now_iso(),
            "base_model_id": CFG["base_model_id"],
            "adapter_dir_used_for_eval": adapter_dir_for_eval,
            "train_columns": sorted(list(CFG["train_columns"])),
            "eval_columns": sorted(list(CFG["eval_columns"])),
            "use_base_for_columns_after_ft": sorted(list(CFG["use_base_for_columns_after_ft"])),
            "lora_targets_used": lora_targets,
            "gen": CFG["gen"],
            "max_new_tokens_by_column": CFG["max_new_tokens_by_column"],
        },
        "base_test_report": base_test_report,
        "lora_test_report": lora_test_report,
        "lora_hybrid_test_report": hybrid_test_report,
        "delta_lora_minus_base": {
            "micro_cer": float(lora_test_report["total_micro"]["cer"] - base_test_report["total_micro"]["cer"]),
            "micro_exact_match": float(lora_test_report["total_micro"]["exact_match"] - base_test_report["total_micro"]["exact_match"]),
            "macro_cer": float(lora_test_report["macro_by_cut"]["macro_cer"] - base_test_report["macro_by_cut"]["macro_cer"]),
            "macro_exact_match": float(lora_test_report["macro_by_cut"]["macro_exact_match"] - base_test_report["macro_by_cut"]["macro_exact_match"]),
            "arrow_only_exact_match": float(lora_test_report["arrow_only_exact_match"] - base_test_report["arrow_only_exact_match"]),
        },
        "delta_hybrid_minus_base": {
            "micro_cer": float(hybrid_test_report["total_micro"]["cer"] - base_test_report["total_micro"]["cer"]),
            "micro_exact_match": float(hybrid_test_report["total_micro"]["exact_match"] - base_test_report["total_micro"]["exact_match"]),
            "macro_cer": float(hybrid_test_report["macro_by_cut"]["macro_cer"] - base_test_report["macro_by_cut"]["macro_cer"]),
            "macro_exact_match": float(hybrid_test_report["macro_by_cut"]["macro_exact_match"] - base_test_report["macro_by_cut"]["macro_exact_match"]),
            "arrow_only_exact_match": float(hybrid_test_report["arrow_only_exact_match"] - base_test_report["arrow_only_exact_match"]),
        },
    }
    summary_path = os.path.join(out_dir, "compare_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {summary_path}")

    print("\n===== FINAL SUMMARY =====")
    print("BASE  total:", base_test_report["total_micro"])
    print("LORA  total:", lora_test_report["total_micro"])
    print("HYBRID total:", hybrid_test_report["total_micro"])
    print("Saved under:", out_dir)


if __name__ == "__main__":
    main()

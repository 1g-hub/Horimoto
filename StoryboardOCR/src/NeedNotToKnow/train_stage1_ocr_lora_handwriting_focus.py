# train_stage1_ocr_lora_handwriting_focus.py
# 
# # Goal: improve handwritten OCR for anime storyboards using LoRA finetuning.
# Key changes vs naive code:
# 1) Training prompt is SHORT (no huge lexicon lists) -> forces visual reading
# 2) Still "single prompt", but includes Target column name (conditioning)
# 3) Handwriting-focused sampling: oversample action_memo/dialogue/picture
# 4) Skip samples whose final_gt is only placeholders (□/△) to avoid wasting steps
# 5) Split by CUT (8:1:1) to avoid leakage

import os
import csv
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # Base model (choose one)
    # "Qwen/Qwen3-VL-4B-Instruct" or "Qwen/Qwen3-VL-8B-Instruct"
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",

    # Data
    "csv_path": "data/episode01/annotation_table.csv",
    "episode_dir": "data/episode01",

    # Which columns to train on (handwriting focus)
    # If you want maximum handwriting gain, you can drop cut/time:
    "train_columns": {"action_memo", "dialogue", "picture", "cut", "time"},

    # Column sampling weights (handwriting focus)
    # Higher -> more frequent in training
    "column_weights": {
        "action_memo": 4.0,
        "dialogue": 4.0,
        "picture": 3.0,
        "cut": 1.0,
        "time": 1.0,
    },

    # Skip rows with empty final_gt
    "skip_empty_gt": True,

    # Skip rows whose final_gt is only placeholders (□/△/spaces/newlines)
    "skip_placeholder_only": True,
    "placeholders": {"□", "△"},

    # Split by CUT (8:1:1)
    "seed": 42,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Output
    "output_dir": "outputs_ft/stage1_ocr_lora_handwriting_focus",

    # Training
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 2,

    "bf16": True,
    "fp16": False,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # Important
    "remove_unused_columns": False,
}
# =========================


# -------------------------
# CSV / split
# -------------------------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_by_cut(rows: List[Dict[str, str]], seed: int, val_ratio: float, test_ratio: float) -> Tuple[List, List, List]:
    cut_ids = sorted({r.get("cut", "") for r in rows if r.get("cut", "")})
    rnd = random.Random(seed)
    rnd.shuffle(cut_ids)

    n = len(cut_ids)
    if n == 0:
        return [], [], []

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
        c = r.get("cut", "")
        if c in val_set:
            val_rows.append(r)
        elif c in test_set:
            test_rows.append(r)
        else:
            train_rows.append(r)
    return train_rows, val_rows, test_rows


def reconstruct_image_path(episode_dir: str, column: str, page: str, row: str) -> str:
    return os.path.join(episode_dir, column, f"page{page}_row{row}.png")


def is_placeholder_only(text: str, placeholders: set) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # remove whitespace/newlines
    t2 = "".join(ch for ch in t if not ch.isspace())
    if not t2:
        return True
    # all chars are placeholders?
    return all(ch in placeholders for ch in t2)


# -------------------------
# Single prompt (short, handwriting-focused)
# -------------------------
def build_single_prompt(column: str) -> str:
    # Single template, but conditioned by column name (still "1 prompt type" conceptually)
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


# -------------------------
# Dataset
# -------------------------
class StoryboardOCRDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], processor: AutoProcessor, episode_dir: str):
        self.processor = processor
        self.episode_dir = episode_dir
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        page = str(r["page"])
        row = str(r["row"])
        col = r["column"]
        gt = (r.get("final_gt") or "").strip()

        img_path = reconstruct_image_path(self.episode_dir, col, page, row)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        prompt = build_single_prompt(col)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
        target_ids = tok(gt, add_special_tokens=False).input_ids
        if tok.eos_token_id is not None:
            target_ids = target_ids + [tok.eos_token_id]

        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_mask = prompt_inputs["attention_mask"][0]
        target_ids_t = torch.tensor(target_ids, dtype=prompt_ids.dtype)

        input_ids = torch.cat([prompt_ids, target_ids_t], dim=0)
        attention_mask = torch.cat([prompt_mask, torch.ones_like(target_ids_t)], dim=0)

        labels = torch.full_like(input_ids, fill_value=-100)
        labels[len(prompt_ids):] = target_ids_t

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = v[0]

        return out


@dataclass
class DataCollatorForVL:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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

        for k in features[0].keys():
            if k in ("input_ids", "attention_mask", "labels"):
                continue
            vals = [f[k] for f in features]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals, dim=0)

        return batch


# -------------------------
# Weighted sampler for handwriting focus
# -------------------------
def build_sample_weights(rows: List[Dict[str, str]]) -> List[float]:
    weights = []
    for r in rows:
        col = r.get("column", "")
        w = float(CFG["column_weights"].get(col, 1.0))
        # optionally boost samples that contain arrows/production cues (cheap heuristic)
        gt = (r.get("final_gt") or "")
        if "→" in gt or "←" in gt or "↑" in gt or "↓" in gt:
            w *= 1.2
        weights.append(w)
    return weights


class WeightedTrainer(Trainer):
    """
    Override get_train_dataloader to use WeightedRandomSampler
    """
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        weights = build_sample_weights(getattr(self.train_dataset, "rows", []))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )


def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    # Load CSV
    rows_all = read_csv_rows(CFG["csv_path"])

    # Filter usable rows
    rows = []
    for r in rows_all:
        col = r.get("column", "")
        if col not in CFG["train_columns"]:
            continue
        if CFG["skip_empty_gt"] and not (r.get("final_gt") or "").strip():
            continue
        if CFG["skip_placeholder_only"] and is_placeholder_only(r.get("final_gt") or "", CFG["placeholders"]):
            continue
        if not r.get("page") or not r.get("row") or not r.get("cut"):
            continue
        rows.append(r)

    if not rows:
        raise RuntimeError("No training samples found after filtering. Check CSV/final_gt.")

    # Split by CUT
    train_rows, val_rows, test_rows = split_by_cut(rows, CFG["seed"], CFG["val_ratio"], CFG["test_ratio"])
    if not train_rows:
        raise RuntimeError("Train split empty. Need more labeled cuts.")

    # Load processor/model
    processor = AutoProcessor.from_pretrained(CFG["model_id"])
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if CFG["bf16"] else None,
    )
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # LoRA
    lora = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    train_ds = StoryboardOCRDataset(train_rows, processor, CFG["episode_dir"])
    eval_ds = StoryboardOCRDataset(val_rows, processor, CFG["episode_dir"]) if val_rows else None

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
        save_steps=CFG["save_steps"],
        save_total_limit=CFG["save_total_limit"],
        bf16=CFG["bf16"],
        fp16=CFG["fp16"],
        report_to="none",
        remove_unused_columns=CFG["remove_unused_columns"],
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=CFG["save_steps"] if eval_ds is not None else None,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(CFG["output_dir"])
    print(f"[OK] saved LoRA adapter to: {CFG['output_dir']}")

    # Save split info
    split_path = os.path.join(CFG["output_dir"], "split_by_cut.json")
    os.makedirs(CFG["output_dir"], exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_cut_ids": sorted({r["cut"] for r in train_rows}),
            "val_cut_ids": sorted({r["cut"] for r in val_rows}),
            "test_cut_ids": sorted({r["cut"] for r in test_rows}),
            "note": "split by CUT to avoid leakage",
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {split_path}")


if __name__ == "__main__":
    main()

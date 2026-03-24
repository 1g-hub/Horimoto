# train_stage1_ocr_lora_qwen3vl_single_prompt.py
# - LoRA finetuning for Qwen3-VL OCR (single shared prompt)
# - Uses the SAME data layout as Stage1:
#     image path: data/<episode>/<column>/page{page}_row{row}.png
# - Uses your annotation CSV:
#     cut,page,row,column,final_gt,...
# - Supports swapping base model 4B/8B by CFG["model_id"]
# - Produces a LoRA adapter dir you can load in stage1_extract_cuts.py with:
#     CFG["use_lora_adapter"]=True
#     CFG["lora_adapter_path"]=<output_dir>

import os
import csv
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

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

    # Train columns (match your stage1 columns; picture is usually not OCR text-heavy but allowed)
    "train_columns": {"cut", "picture", "action_memo", "dialogue", "time"},
    "skip_empty_gt": True,

    # Split by CUT (no leakage) : train/val/test = 8/1/1
    "seed": 42,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Prompt resources (same as stage1)
    "script_phase": "script_phase1",
    "episode_id": "episode01",
    "symbol_lexicon_path": "data/storyboard_symbol_lexicon.json",
    "lexicon_path": None,  # under outputs/<script_phase>/<episode_id>/

    "max_symbol_terms": 400,
    "max_char_names": 200,

    # Output
    "output_dir": "outputs_ft/stage1_ocr_lora_qwen3vl_single_prompt",

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

    # Precision
    "bf16": True,
    "fp16": False,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # Important for vision inputs
    "remove_unused_columns": False,
}
# =========================


# -------------------------
# JSON / CSV loaders
# -------------------------
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def reconstruct_image_path(episode_dir: str, column: str, page: str, row: str) -> str:
    # data/episode01/<column>/page{page}_row{row}.png
    return os.path.join(episode_dir, column, f"page{page}_row{row}.png")


# -------------------------
# Prompt building (single shared prompt)
# -------------------------
def load_storyboard_symbol_words(path: str, max_terms: int) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    data = read_json(path)
    words = []
    seen = set()
    if isinstance(data, list):
        for item in data:
            w = item.get("word")
            if isinstance(w, list):
                for s in w:
                    if isinstance(s, str):
                        t = s.strip()
                        if t and t not in seen:
                            seen.add(t)
                            words.append(t)
            elif isinstance(w, str):
                t = w.strip()
                if t and t not in seen:
                    seen.add(t)
                    words.append(t)
    words.sort(key=lambda x: (-len(x), x))
    return words[:max_terms]

def load_lexicon_characters(path_json: str, max_names: int) -> List[str]:
    if not path_json or not os.path.isfile(path_json):
        return []
    obj = read_json(path_json)
    chars = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            name = e.get("canonical")
            if isinstance(name, str) and name.strip():
                chars.append(name.strip())
    # add short forms
    out = []
    seen = set()
    for n in sorted(chars, key=lambda x: (-len(x), x)):
        if n not in seen:
            seen.add(n); out.append(n)
        if "・" in n:
            short = n.split("・")[0]
            if short and short not in seen:
                seen.add(short); out.append(short)
    return out[:max_names]

def build_single_ocr_prompt(symbol_words: List[str], lexicon_characters: List[str]) -> str:
    glossary = ""
    if symbol_words:
        glossary = "=== STORYBOARD TERMS (terms only) ===\n" + "\n".join(symbol_words) + "\n\n"

    chars = ""
    if lexicon_characters:
        chars = "=== CHARACTER NAMES (terms only) ===\n" + "\n".join(lexicon_characters) + "\n\n"

    prompt = (
        "You are an OCR engine for anime storyboards.\n"
        "This storyboard contains BOTH handwritten text and computer-typed text.\n"
        "Handwritten text may include arrows and symbols.\n\n"
        "Task: transcribe ALL visible text as faithfully as possible.\n"
        "Rules:\n"
        "- Keep original line breaks.\n"
        "- Preserve symbols exactly (e.g., arrows → ← ↑ ↓, brackets, punctuation).\n"
        "- Do NOT correct typos.\n"
        "- Do NOT translate.\n"
        "- If unreadable, output '□'.\n"
        "- Return ONLY the transcribed text. No explanations.\n\n"
        "Handwriting notes:\n"
        "- Arrows are important. Always transcribe arrows if present.\n"
        "- Production terms may be handwritten (e.g., PAN, QPAN, follow, back, FIX, IN, OUT).\n"
        "- If a handwritten token looks like a known term, prefer the known term spelling exactly.\n"
        "- If arrows are attached to a term (e.g., →PAN), keep them together as seen.\n\n"
        f"{chars}"
        f"{glossary}"
        "Important:\n"
        "- Prefer transcribing production terms/symbols and character names exactly.\n"
        "- Do NOT infer meaning; only transcribe what you see.\n"
    )
    return prompt


# -------------------------
# Split by CUT (8:1:1) without leakage
# -------------------------
def split_by_cut(rows: List[Dict[str, str]], seed: int, val_ratio: float, test_ratio: float) -> Tuple[List, List, List]:
    cut_ids = sorted({r.get("cut", "") for r in rows if r.get("cut", "")})
    rnd = random.Random(seed)
    rnd.shuffle(cut_ids)

    n = len(cut_ids)
    if n == 0:
        return [], [], []

    n_val = max(1, int(n * val_ratio)) if n >= 10 else max(1, int(n * val_ratio)) if n >= 3 else 0
    n_test = max(1, int(n * test_ratio)) if n >= 10 else max(1, int(n * test_ratio)) if n >= 3 else 0

    # keep within bounds
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


# -------------------------
# Dataset (vision-language supervised)
# -------------------------
class StoryboardOCRDataset(Dataset):
    """
    Each sample = (image + prompt) -> target text(final_gt)

    Implementation detail:
    - Build prompt tokens with apply_chat_template(add_generation_prompt=True)
    - Append target tokens
    - labels mask prompt part with -100
    """
    def __init__(self, rows: List[Dict[str, str]], processor: AutoProcessor, prompt: str, episode_dir: str):
        self.processor = processor
        self.prompt = prompt
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

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt},
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

        # vision inputs
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

        # stack vision keys if present
        for k in features[0].keys():
            if k in ("input_ids", "attention_mask", "labels"):
                continue
            vals = [f[k] for f in features]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals, dim=0)

        return batch


def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    # Load CSV
    rows_all = read_csv_rows(CFG["csv_path"])

    # Filter usable rows (column + final_gt)
    rows = []
    for r in rows_all:
        col = r.get("column", "")
        if col not in CFG["train_columns"]:
            continue
        if CFG["skip_empty_gt"] and not (r.get("final_gt") or "").strip():
            continue
        # must have page/row/cut
        if not r.get("page") or not r.get("row") or not r.get("cut"):
            continue
        rows.append(r)

    if not rows:
        raise RuntimeError("No training samples found. Check CSV/final_gt/train_columns.")

    # Split by CUT (no leakage)
    train_rows, val_rows, test_rows = split_by_cut(rows, CFG["seed"], CFG["val_ratio"], CFG["test_ratio"])
    if not train_rows:
        raise RuntimeError("Train split is empty. Need more labeled cuts.")
    if not val_rows:
        print("[WARN] val split is empty (too few cuts). You can still train.")
    if not test_rows:
        print("[WARN] test split is empty (too few cuts).")

    # Load processor/model
    processor = AutoProcessor.from_pretrained(CFG["model_id"])

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if CFG["bf16"] else None,
    )
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # LoRA wrap
    lora = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    # Build single shared prompt from the same resources as stage1
    symbol_words = load_storyboard_symbol_words(CFG["symbol_lexicon_path"], CFG["max_symbol_terms"])

    lexicon_json = os.path.join("outputs", CFG["script_phase"], CFG["episode_id"], CFG["lexicon_path"])
    lexicon_characters = load_lexicon_characters(lexicon_json, CFG["max_char_names"])

    prompt = build_single_ocr_prompt(symbol_words, lexicon_characters)

    # Datasets
    train_ds = StoryboardOCRDataset(train_rows, processor, prompt, CFG["episode_dir"])
    eval_ds = StoryboardOCRDataset(val_rows, processor, prompt, CFG["episode_dir"]) if val_rows else None

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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save LoRA adapter
    trainer.save_model(CFG["output_dir"])
    print(f"[OK] saved LoRA adapter to: {CFG['output_dir']}")

    # (Optional) save split info for reproducibility
    split_path = os.path.join(CFG["output_dir"], "split_by_cut.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_cut_ids": sorted({r["cut"] for r in train_rows}),
            "val_cut_ids": sorted({r["cut"] for r in val_rows}),
            "test_cut_ids": sorted({r["cut"] for r in test_rows}),
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {split_path}")


if __name__ == "__main__":
    main()

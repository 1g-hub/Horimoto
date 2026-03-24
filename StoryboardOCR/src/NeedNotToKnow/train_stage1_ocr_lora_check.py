# train_stage1_ocr_lora_full.py
import os
import csv
import json
import random
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer, TrainerCallback
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT HERE)
# =========================
CFG = {
    # Base model: switch 4B / 8B here
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    # "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    # "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    # "model_id": "Qwen/Qwen3-VL-32B-Instruct",

    # Data
    "csv_path": "data/episode01/annotation_table.csv",
    "episode_dir": "data/episode01",

    # Which columns to use
    "train_columns": {"action_memo", "dialogue", "picture"},  # {"action_memo", "dialogue", "picture", "cut", "time"}

    # Optional: oversample handwriting-heavy columns
    "use_weighted_sampling": True,
    "column_weights": {
        "action_memo": 4.0,
        "dialogue": 4.0,
        "picture": 3.0,
        "cut": 1.0,
        "time": 1.0,
    },

    # Data filters
    "skip_empty_gt": True,
    "skip_placeholder_only": True,
    "placeholders": {"□", "△"},

    # Selection by page/cut (either can be used)
    "select_pages": False,
    "page_start": 1,
    "page_end": 9999,

    "select_cuts": True,
    "cut_start": 1,
    "cut_end": 130,

    # Split by CUT (8:1:1)
    "seed": 42,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Training
    "output_dir": "outputs_ft/stage1_ocr_lora_full",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "logging_steps": 1,
    "save_steps": 10,
    "save_total_limit": 2,

    # Precision
    "bf16": True,
    "fp16": False,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # Eval (CER/ExactMatch) settings
    "neutral_symbols": ["□", "△"],      # removed before CER/ExactMatch
    "eval_space_insensitive": True,     # collapse whitespace for metrics
    "eval_max_new_tokens": 256,
    "eval_max_samples": 200,            # per split evaluation cap

    # Run test evaluation or not
    "run_test": False,

    # Plot
    "plot_loss_path": "loss_curve.png",
}
# =========================


# -------------------------
# CSV reading assumptions
# -------------------------
REQUIRED_COLUMNS = {"cut", "page", "row", "column", "final_gt"}

def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = REQUIRED_COLUMNS - set(rows[0].keys()) if rows else REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return rows

def reconstruct_image_path(episode_dir: str, column: str, page: str, row: str) -> str:
    # Must match Stage1 crops directory structure
    return os.path.join(episode_dir, column, f"page{page}_row{row}.png")

def is_placeholder_only(text: str, placeholders: set) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    t2 = "".join(ch for ch in t if not ch.isspace())
    if not t2:
        return True
    return all(ch in placeholders for ch in t2)

def in_page_range(page: int) -> bool:
    if not CFG["select_pages"]:
        return True
    return CFG["page_start"] <= page <= CFG["page_end"]

def in_cut_range(cut: int) -> bool:
    if not CFG["select_cuts"]:
        return True
    return CFG["cut_start"] <= cut <= CFG["cut_end"]


# -------------------------
# Split by CUT (no leakage)
# -------------------------
def split_by_cut(rows: List[Dict[str, str]], seed: int, val_ratio: float, test_ratio: float) -> Tuple[List, List, List]:
    cut_ids = sorted({r["cut"] for r in rows if r.get("cut")})
    rnd = random.Random(seed)
    rnd.shuffle(cut_ids)

    n = len(cut_ids)
    if n == 0:
        return [], [], []

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    # keep at least 1 if enough cuts
    if n >= 10:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    # avoid overflow
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
    return train_rows, val_rows, test_rows


# -------------------------
# Prompt (1個。列名で条件付けするが“単一テンプレ”)
# -------------------------
def build_prompt(column: str) -> str:
    return (
        "You are an OCR engine for anime storyboards.\n"
        f"Target column: {column}.\n"
        "This storyboard contains BOTH handwritten and computer-typed text.\n"
        "Handwritten text often includes arrows and short production terms.\n\n"
        "Task: transcribe ALL visible text as faithfully as possible.\n"
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
        self.rows = rows
        self.processor = processor
        self.episode_dir = episode_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        col = r["column"]
        page = r["page"]
        row = r["row"]
        gt = (r.get("final_gt") or "").strip()

        img_path = reconstruct_image_path(self.episode_dir, col, page, row)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")

        prompt = build_prompt(col)

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

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # vision tensors
        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = v[0]
        return out


def _shape(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    return f"type={type(x).__name__}"


@dataclass
class DataCollatorForVLDebug:
    """
    Debug/Assert版 DataCollator for Qwen3-VL 系。

    - 返すバッチは基本「Tensorのみ」に限定（非Tensorは捨てる or assert）
    - 必須キー: input_ids / attention_mask / labels
    - 画像系キー (pixel_values / image_grid_thw 等) が入っているか確認
    - モデル forward のシグネチャから「未知キー」を警告/エラーにできる（strict=True）
    """
    processor: AutoProcessor
    model: Optional[torch.nn.Module] = None

    # strict=True なら forward に無いキーをエラーにする（ただし forward が **kwargs の場合は判定不能）
    strict: bool = True

    # 最初のNバッチだけ詳細表示
    print_first_n: int = 2

    # 非Tensorキーが入ってたらエラーにする（Falseなら黙って捨てる）
    assert_no_non_tensor: bool = True

    # これらのキーは「入ってたら危険」(メタ情報など)
    denylist_keys: Set[str] = field(default_factory=lambda: {
        "cut", "page", "row", "ref_text", "column_name", "text", "meta"
    })

    # 画像入力としてよく出るキー（環境により多少違う）
    expected_vision_keys_any_of: Set[str] = field(default_factory=lambda: {
        "pixel_values",
        "image_grid_thw",
        "video_grid_thw",
        "pixel_values_videos",
    })

    _call_count: int = 0

    def _forward_accepts_kwargs(self) -> bool:
        if self.model is None:
            return True
        try:
            sig = inspect.signature(self.model.forward)
        except Exception:
            # signatureが取れないなら判定不能なのでkwargs扱いにする
            return True
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def _forward_param_names(self) -> Set[str]:
        if self.model is None:
            return set()
        try:
            sig = inspect.signature(self.model.forward)
            return set(sig.parameters.keys())
        except Exception:
            return set()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        self._call_count += 1
        tok = self.processor.tokenizer

        if not features:
            raise ValueError("Empty features in collator.")

        # -------------------------
        # 0) Feature key sanity
        # -------------------------
        required = {"input_ids", "attention_mask", "labels"}
        for i, f in enumerate(features):
            missing = required - set(f.keys())
            if missing:
                raise KeyError(f"[Collator] feature[{i}] missing keys: {sorted(missing)}")

            # denylist（メタ情報）が混ざってないか
            bad = set(f.keys()) & self.denylist_keys
            if bad:
                # これは「モデルに渡さないつもり」ならOKだが、混ざるなら検出したい
                # assert_no_non_tensor と同様、まずは警告/エラーに寄せる
                msg = f"[Collator] feature[{i}] contains denylist keys: {sorted(bad)}"
                if self.assert_no_non_tensor:
                    raise ValueError(msg)
                else:
                    print("WARN:", msg)

        # -------------------------
        # 1) pad input_ids / attention_mask / labels
        # -------------------------
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 1D前提（Dataset側で 1D にしている想定）
        # もし 2D/3D が混ざったら事故なので assert
        for name, seqs in [("input_ids", input_ids), ("attention_mask", attention_mask), ("labels", labels)]:
            for i, t in enumerate(seqs):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"[Collator] {name}[{i}] is not Tensor: {type(t)}")
                if t.dim() != 1:
                    raise ValueError(f"[Collator] {name}[{i}] must be 1D, got {t.dim()}D {tuple(t.shape)}")

        batch = tok.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full(
            (len(labels), max_len),
            fill_value=-100,
            dtype=batch["input_ids"].dtype,
        )
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        batch["labels"] = padded_labels

        # -------------------------
        # 2) その他キー（vision系など）を stack
        #    - Tensorだけを batch に残す
        # -------------------------
        extra_keys = sorted(set().union(*[set(f.keys()) for f in features]) - {"input_ids", "attention_mask", "labels"})
        dropped_non_tensor = []
        stacked_tensor_keys = []

        for k in extra_keys:
            vals = [f.get(k, None) for f in features]

            # 欠けてるfeatureがあるなら事故りやすいので assert（揃ってるべき）
            if any(v is None for v in vals):
                raise KeyError(f"[Collator] key '{k}' missing in some features. Fix dataset __getitem__ to always provide it.")

            # 非Tensorが混ざるなら捨てる/エラー
            if not all(isinstance(v, torch.Tensor) for v in vals):
                dropped_non_tensor.append(k)
                if self.assert_no_non_tensor:
                    types = [type(v).__name__ for v in vals]
                    raise TypeError(f"[Collator] Non-tensor key '{k}' in batch. types={types}")
                continue

            # Tensorならstack（shape不一致も事故なので assert）
            shape0 = tuple(vals[0].shape)
            if any(tuple(v.shape) != shape0 for v in vals):
                shapes = [tuple(v.shape) for v in vals]
                raise ValueError(f"[Collator] Tensor key '{k}' has inconsistent shapes: {shapes}")

            batch[k] = torch.stack(vals, dim=0)
            stacked_tensor_keys.append(k)

        # -------------------------
        # 3) 「画像系キーが存在するか」チェック
        # -------------------------
        has_any_vision = any(k in batch for k in self.expected_vision_keys_any_of)
        if not has_any_vision:
            # 画像を見ていない可能性が高いので強く警告
            msg = (
                "[Collator] No vision keys found in batch. "
                f"Expected any of {sorted(self.expected_vision_keys_any_of)}. "
                "If this is Stage1 OCR training, this is usually a BUG (images not fed)."
            )
            # ここは好みだが、まずはエラーにして気づけるようにするのがおすすめ
            raise RuntimeError(msg)

        # -------------------------
        # 4) strict: forward で受け取れないキーを検査（可能な場合のみ）
        # -------------------------
        if self.model is not None and self.strict:
            if not self._forward_accepts_kwargs():
                forward_keys = self._forward_param_names()
                unknown = set(batch.keys()) - forward_keys
                if unknown:
                    raise ValueError(
                        "[Collator] Batch contains keys not in model.forward signature:\n"
                        f"  unknown={sorted(unknown)}\n"
                        f"  forward_params={sorted(forward_keys)}"
                    )
            # forward が **kwargs の場合は判定不能なので print だけにする
            else:
                # ここで何もしない（printで見たい場合は下のprintに出る）
                pass

        # -------------------------
        # 5) print（最初のN回だけ）
        # -------------------------
        if self._call_count <= self.print_first_n:
            print("\n" + "=" * 80)
            print(f"[CollatorDebug] call={self._call_count}  batch_size={len(features)}")
            print("[CollatorDebug] batch keys:", sorted(batch.keys()))
            for k in sorted(batch.keys()):
                print("  ", k, "->", _shape(batch[k]))
            if dropped_non_tensor:
                print("[CollatorDebug] dropped non-tensor keys:", dropped_non_tensor)
            print("[CollatorDebug] stacked tensor keys:", stacked_tensor_keys)
            print("=" * 80 + "\n")

        return batch


# -------------------------
# Weighted sampling (optional)
# -------------------------
def build_sample_weights(rows: List[Dict[str, str]]) -> List[float]:
    weights = []
    for r in rows:
        col = r.get("column", "")
        w = float(CFG["column_weights"].get(col, 1.0))
        gt = r.get("final_gt", "") or ""
        # small bonus for arrows
        if "→" in gt or "←" in gt or "↑" in gt or "↓" in gt:
            w *= 1.2
        weights.append(w)
    return weights

class WeightedTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if not CFG["use_weighted_sampling"]:
            return super().get_train_dataloader()
        weights = build_sample_weights(getattr(self.train_dataset, "rows", []))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )


# -------------------------
# CER / ExactMatch evaluation (base on generation)
# -------------------------
def normalize_for_eval(s: str, neutral_symbols: List[str]) -> str:
    s = s or ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for sym in neutral_symbols:
        s = s.replace(sym, "")
    if CFG["eval_space_insensitive"]:
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
    hyp2 = normalize_for_eval(hyp, CFG["neutral_symbols"])
    ref2 = normalize_for_eval(ref, CFG["neutral_symbols"])
    h = list(hyp2)
    r = list(ref2)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return levenshtein(h, r) / len(r)

def exact_match(hyp: str, ref: str) -> int:
    return int(normalize_for_eval(hyp, CFG["neutral_symbols"]) == normalize_for_eval(ref, CFG["neutral_symbols"]))


def _ensure_2d_ids(x: torch.Tensor) -> torch.Tensor:
    """
    Qwen3-VL が期待する (batch, seq_len) へ正規化する。
    ありがちな崩れ:
      - (seq_len,) -> (1, seq_len)
      - (batch, 1, seq_len) -> (batch, seq_len)
    """
    if x is None:
        return x
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 3 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def _move_and_fix_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v

    # ★ここが本質
    if "input_ids" in out and isinstance(out["input_ids"], torch.Tensor):
        out["input_ids"] = _ensure_2d_ids(out["input_ids"])
    if "attention_mask" in out and isinstance(out["attention_mask"], torch.Tensor):
        out["attention_mask"] = _ensure_2d_ids(out["attention_mask"])

    # もし vision 系にも (batch,1,...) が混ざるなら同様に潰す（保険）
    for k in ["image_grid_thw"]:
        if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() == 3 and out[k].size(1) == 1:
            out[k] = out[k].squeeze(1)

    return out


def _set_use_cache_false_everywhere(model):
    # peft wrapper / base model 両方に効かせる
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        # PeftModelの場合
        if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = False
    except Exception:
        pass
    try:
        # さらに深い場合（環境差）
        if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "config"):
            model.base_model.model.config.use_cache = False
    except Exception:
        pass


def _disable_gradient_checkpointing_for_eval(model):
    """
    generate/eval 中の warning と不整合を減らす。
    学習が終わった後の評価なら、切りっぱなしでOK。
    """
    # PEFTの場合 base を掘る
    base = model
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model"):
            base = base.model

    # transformers系で一般的なAPI
    if hasattr(base, "gradient_checkpointing_disable"):
        try:
            base.gradient_checkpointing_disable()
        except Exception:
            pass


@torch.no_grad()
def _greedy_generate_no_cache(model, input_ids, attention_mask, extra_kwargs, max_new_tokens, eos_token_id):
    """
    generate() を使わず、毎ステップ full input を model.forward(use_cache=False) に渡す。
    これで input_ids と attention_mask の長さは常に一致し、Qwen3-VLのrope計算が壊れにくい。
    """
    for _ in range(max_new_tokens):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **extra_kwargs,
        )
        # 次トークンは greedy
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)

        if eos_token_id is not None:
            # 全バッチがEOSなら終了（今回B=1想定）
            if torch.all(next_id.squeeze(1) == eos_token_id):
                break

    return input_ids


@torch.no_grad()
def generate_hyp(model, processor, col: str, img: Image.Image) -> str:
    # ★評価前にキャッシュ無効化＆GC無効化（毎回呼んでOK）
    _set_use_cache_false_everywhere(model)
    _disable_gradient_checkpointing_for_eval(model)

    prompt = build_prompt(col)

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
    batch = _move_and_fix_batch(batch, model.device)

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    # generateへ渡す追加入力（pixel_values等）
    extra = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)

    # 念のため attention_mask が無い場合は作る
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    prompt_len = input_ids.shape[1]

    # まずは generate(use_cache=False) を試し、落ちたら自前greedyへフォールバック
    try:
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_id,
            max_new_tokens=CFG["eval_max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=False,          # ★最重要
            **extra,
        )[0]
    except IndexError as e:
        # ★ここであなたのエラーを吸収する
        # print(f"[WARN] generate failed, fallback to greedy_no_cache: {e}")
        out_full = _greedy_generate_no_cache(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            extra_kwargs=extra,
            max_new_tokens=CFG["eval_max_new_tokens"],
            eos_token_id=eos_id,
        )
        out_ids = out_full[0]

    hyp = tok.decode(out_ids[prompt_len:], skip_special_tokens=True).strip()
    return hyp


def eval_split(model, processor, rows: List[Dict[str, str]], name: str) -> Dict[str, Any]:
    # sample cap
    rows2 = rows[:CFG["eval_max_samples"]] if CFG["eval_max_samples"] else rows
    total_cer = 0.0
    total_em = 0
    n = 0
    by_col: Dict[str, Dict[str, Any]] = {}

    for r in tqdm(rows2, desc=f"Eval {name}", leave=False):
        col = r["column"]
        img_path = reconstruct_image_path(CFG["episode_dir"], col, r["page"], r["row"])
        if not os.path.isfile(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        hyp = generate_hyp(model, processor, col, img)
        ref = r["final_gt"]

        c = cer(hyp, ref)
        e = exact_match(hyp, ref)

        total_cer += c
        total_em += e
        n += 1

        if col not in by_col:
            by_col[col] = {"n": 0, "cer_sum": 0.0, "em_sum": 0}
        by_col[col]["n"] += 1
        by_col[col]["cer_sum"] += c
        by_col[col]["em_sum"] += e

    out = {
        "name": name,
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


# -------------------------
# Loss plotting (train/val)
# -------------------------
def plot_loss_curve(log_history: List[Dict[str, Any]], out_path: str):
    # Trainer logs: train loss under "loss", eval loss under "eval_loss"
    steps_train, loss_train = [], []
    steps_val, loss_val = [], []

    for item in log_history:
        if "loss" in item and "step" in item and "eval_loss" not in item:
            steps_train.append(item["step"])
            loss_train.append(item["loss"])
        if "eval_loss" in item and "step" in item:
            steps_val.append(item["step"])
            loss_val.append(item["eval_loss"])

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


def infer_lora_target_modules(model) -> list[str]:
    """
    Model内の Linear 層名から、LoRAを刺す対象を自動推定する。
    典型候補（q_proj等）が存在する場合のみ採用する。
    """
    candidate_names = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in candidate_names:
                found.add(leaf)
    # 何も見つからない場合は、よくあるattention/MLP名を試す（環境依存の保険）
    if not found:
        # ここに出る場合は model が特殊。とりあえず一般的な候補を返す
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return sorted(found)


# -------------------------
# Main
# -------------------------
def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    rows_all = read_csv_rows(CFG["csv_path"])

    # Filter to match annotation_table.csv assumptions + selection conditions
    rows = []
    for r in rows_all:
        col = r.get("column", "")
        if col not in CFG["train_columns"]:
            continue
        if CFG["skip_empty_gt"] and not (r.get("final_gt") or "").strip():
            continue
        if CFG["skip_placeholder_only"] and is_placeholder_only(r.get("final_gt") or "", CFG["placeholders"]):
            continue
        if not r.get("cut") or not r.get("page") or not r.get("row"):
            continue

        cut_i = int(r["cut"])
        page_i = int(r["page"])
        if not in_cut_range(cut_i):
            continue
        if not in_page_range(page_i):
            continue

        rows.append(r)

    if not rows:
        raise RuntimeError("No usable rows after filtering. Check ranges & CSV.")

    train_rows, val_rows, test_rows = split_by_cut(rows, CFG["seed"], CFG["val_ratio"], CFG["test_ratio"])
    if not train_rows:
        raise RuntimeError("Train split is empty. Need more labeled cuts or loosen filters.")

    os.makedirs(CFG["output_dir"], exist_ok=True)

    # Save split ids (for later baseline vs LoRA comparison)
    split_path = os.path.join(CFG["output_dir"], "split_by_cut.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_cut_ids": sorted({r["cut"] for r in train_rows}),
            "val_cut_ids": sorted({r["cut"] for r in val_rows}),
            "test_cut_ids": sorted({r["cut"] for r in test_rows}),
            "filters": {
                "select_pages": CFG["select_pages"],
                "page_start": CFG["page_start"],
                "page_end": CFG["page_end"],
                "select_cuts": CFG["select_cuts"],
                "cut_start": CFG["cut_start"],
                "cut_end": CFG["cut_end"],
                "train_columns": sorted(list(CFG["train_columns"])),
                "skip_placeholder_only": CFG["skip_placeholder_only"],
            }
        }, f, ensure_ascii=False, indent=2)

    # Load model
    processor = AutoProcessor.from_pretrained(CFG["model_id"])
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CFG["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if CFG["bf16"] else None,
    )
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # --- LoRA wrap ---
    target_modules = infer_lora_target_modules(base_model)

    lora = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,  # ★必須
    )

    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()
    print("[LoRA] target_modules =", target_modules)

    train_ds = StoryboardOCRDataset(train_rows, processor, CFG["episode_dir"])
    val_ds = StoryboardOCRDataset(val_rows, processor, CFG["episode_dir"]) if val_rows else None
    test_ds = StoryboardOCRDataset(test_rows, processor, CFG["episode_dir"]) if test_rows else None

    collator = DataCollatorForVLDebug(
        processor=processor,
        model=model,           # ★ここ重要（LoRA wrapped modelでもOK）
        strict=True,
        print_first_n=2,
        assert_no_non_tensor=True,
    )

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
        remove_unused_columns=False,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=CFG["save_steps"] if val_ds is not None else None,
    )

    trainer_cls = WeightedTrainer if CFG["use_weighted_sampling"] else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    dl = DataLoader(train_ds, batch_size=CFG["per_device_train_batch_size"], collate_fn=collator)
    batch = next(iter(dl))  # ここで collator の print / assert が走る

    # ついでに forward が通るか確認（重いので1回だけ）
    batch_gpu = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch_gpu)
    print("[OK] one forward pass works. loss=", float(out.loss))

    trainer.train()

    # Save adapter
    trainer.save_model(CFG["output_dir"])

    # Save log history + loss plot
    log_history = trainer.state.log_history
    log_path = os.path.join(CFG["output_dir"], "log_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    loss_path = os.path.join(CFG["output_dir"], CFG["plot_loss_path"])
    plot_loss_curve(log_history, loss_path)

    # Optional: evaluate (val/test) with CER/ExactMatch using generation
    # Note: evaluation here is to check training is meaningful.
    metrics_out = {}
    if val_rows:
        metrics_out["val"] = eval_split(model, processor, val_rows, "val")
    if CFG["run_test"] and test_rows:
        metrics_out["test"] = eval_split(model, processor, test_rows, "test")

    metrics_path = os.path.join(CFG["output_dir"], "metrics_post_train.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved adapter to: {CFG['output_dir']}")
    print(f"[OK] wrote split: {split_path}")
    print(f"[OK] wrote loss plot: {loss_path}")
    print(f"[OK] wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()

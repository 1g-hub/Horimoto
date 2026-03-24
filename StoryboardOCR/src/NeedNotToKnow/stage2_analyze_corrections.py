# stage2_analyze_corrections_from_csv.py
#
# Input CSV header (REQUIRED):
#   cut,page,row,column,stage1_ocr,stage2_corrected,final_gt,error_type,note
#
# This script analyzes Stage2 correction behavior:
# - number of changed samples
# - improved / worsened / neutral (by CER)
# - exact match fixed / broken
# - character-name correction counts (lexicon-based; substring match)
# - dialogue correction counts (quote-based and full-text CER)
# - lists of how text was corrected (diff span)
# - ratios: correct correction rate / wrong correction rate
#
# Outputs:
#   /app/outputs_eval/<episode>_stage2_correction_analysis_<variant>.json
#   /app/outputs_eval/<episode>_stage2_correction_analysis_<variant>.details.csv

import os
import re
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Iterable


# ==================================================
# CONFIG (EDIT HERE)
# ==================================================
APP_ROOT = "/app"
EPISODE_ID = "episode01"

# ★あなたの「評価用CSV」（このヘッダ形式のやつ）を指すパスに変更して使ってください
IN_CSV = f"{APP_ROOT}/data/{EPISODE_ID}/annotation_table.csv"

# キャラ名辞書（任意：無ければキャラ名分析はスキップ）
LEXICON_PATH = f"{APP_ROOT}/outputs/script_phase1/{EPISODE_ID}/lexicon_entities.json"

# 出力
OUT_DIR = f"{APP_ROOT}/outputs_stage2_clm"
STAGE2_VARIANT = "nojp"  # ファイル名につけるだけ（csv内にvariantが無くてもOK）
OUT_JSON = os.path.join(OUT_DIR, f"{EPISODE_ID}_stage2_correction_analysis_{STAGE2_VARIANT}.json")
OUT_DETAILS_CSV = os.path.join(OUT_DIR, f"{EPISODE_ID}_stage2_correction_analysis_{STAGE2_VARIANT}.details.csv")

CFG = {
    # 解析対象列（csvの "column" の値に合わせる）
    # stage2が実際に触る列だけ見たいなら ["action_memo","dialogue"] にする
    "eval_columns": ["action_memo", "dialogue"],  # ["action_memo", "dialogue", "cut", "time", "picture"]

    # □/△ をGT/予測/入力から消して評価する（あなたの既存方針に合わせる）
    "neutral_symbols_removed": ["□", "△"],

    # 文字正規化
    "collapse_spaces": True,
    "keep_newlines": True,

    # CER閾値（改善/悪化判定の微小誤差）
    "cer_eps": 1e-12,

    # セリフ評価：カギ括弧「」の中身を別評価する
    "quote_regex": r"「([^」]+)」",

    # 出力CSVで長文が邪魔なら短くする
    "max_text_len_in_details": 600,

    # stage2_corrected が空のときの扱い
    # True: stage2_corrected="" は stage1_ocr と同一扱い（=修正無し）
    # False: 空は「欠損」として扱う
    "treat_empty_stage2_as_stage1": True,
}
# ==================================================


REQUIRED_HEADER = [
    "cut", "page", "row", "column",
    "stage1_ocr", "stage2_corrected", "final_gt",
    "error_type", "note"
]


# -------------------------
# Utilities
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def normalize_column_name(s: str) -> str:
    """
    CSV column values -> internal field names.
    """
    if not s:
        return ""
    t = str(s).strip().lower()
    t = t.replace("　", " ")
    t = re.sub(r"\s+", "", t)
    t = t.replace("-", "_")

    alias = {
        "action": "action_memo",
        "actionmemo": "action_memo",
        "memo": "action_memo",
        "action_memo": "action_memo",

        "dialog": "dialogue",
        "dialogue": "dialogue",
        "serif": "dialogue",
        "speech": "dialogue",

        "picture": "picture",
        "time": "time",
        "cut": "cut",
    }
    return alias.get(t, t)


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")

    for sym in CFG["neutral_symbols_removed"]:
        s = s.replace(sym, "")

    if CFG["keep_newlines"]:
        lines = [ln.strip() for ln in s.split("\n")]
        lines = [ln for ln in lines if ln != ""]
        s = "\n".join(lines)
    else:
        s = " ".join([ln.strip() for ln in s.split("\n") if ln.strip() != ""])

    if CFG["collapse_spaces"]:
        s = re.sub(r"[ \t]+", " ", s)

    return s.strip()


def clip_text(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...(trunc)"


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    a = a or ""
    b = b or ""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def compute_cer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref = ref or ""
    hyp = hyp or ""
    if len(ref) == 0:
        if hyp == "":
            return 0.0, 0, 0
        return 1.0, len(hyp), 0
    ed = levenshtein_distance(ref, hyp)
    return ed / max(len(ref), 1), ed, len(ref)


def compute_diff_span(a: str, b: str) -> Dict[str, Any]:
    if a == b:
        return {"changed": False}

    i = 0
    min_len = min(len(a), len(b))
    while i < min_len and a[i] == b[i]:
        i += 1

    j = 0
    a_rem = a[i:]
    b_rem = b[i:]
    min_rem = min(len(a_rem), len(b_rem))
    while j < min_rem and a_rem[-(j + 1)] == b_rem[-(j + 1)]:
        j += 1

    a_start = i
    a_end = len(a) - j
    b_start = i
    b_end = len(b) - j

    return {
        "changed": True,
        "a_span": [a_start, max(a_start, a_end)],
        "b_span": [b_start, max(b_start, b_end)],
        "from_sub": a[a_start:a_end],
        "to_sub": b[b_start:b_end],
    }


def extract_quotes(text: str) -> str:
    if not text:
        return ""
    qs = re.findall(CFG["quote_regex"], text)
    qs = [q.strip() for q in qs if q and q.strip()]
    return "\n".join(qs).strip()


def sniff_csv_dialect(path: str) -> csv.Dialect:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            return dialect
        except Exception:
            return csv.excel


def load_eval_csv(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"IN_CSV not found: {path}")

    dialect = sniff_csv_dialect(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        header = reader.fieldnames or []

        missing = [h for h in REQUIRED_HEADER if h not in header]
        if missing:
            raise ValueError(
                "CSV header mismatch.\n"
                f"Required: {REQUIRED_HEADER}\n"
                f"Got: {header}\n"
                f"Missing: {missing}"
            )

        rows = list(reader)

    meta = {
        "path": path,
        "delimiter": getattr(dialect, "delimiter", ","),
        "n_rows": len(rows),
        "header": header,
    }
    return rows, meta


# -------------------------
# Character lexicon (optional)
# -------------------------
def load_character_names(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    obj = read_json(path)
    names = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            n = e.get("canonical")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())

    # add short forms for "A・B"
    seen = set()
    out = []
    for n in sorted(names, key=lambda x: (-len(x), x)):
        if n not in seen:
            seen.add(n)
            out.append(n)
        if "・" in n:
            short = n.split("・")[0]
            if short and short not in seen:
                seen.add(short)
                out.append(short)
    return out


# -------------------------
# Aggregation
# -------------------------
@dataclass
class Agg:
    n: int = 0

    sum_ref_len: int = 0
    sum_ed_stage1: int = 0
    sum_ed_stage2: int = 0

    sum_cer_stage1: float = 0.0
    sum_cer_stage2: float = 0.0

    sum_em_stage1: int = 0
    sum_em_stage2: int = 0

    changed: int = 0
    improved: int = 0
    worsened: int = 0
    neutral_changed: int = 0

    fixed_em: int = 0
    broke_em: int = 0

    def update(self, ref: str, s1: str, s2: str, cer1: float, cer2: float, ed1: int, ed2: int, ref_len: int):
        self.n += 1
        self.sum_ref_len += ref_len
        self.sum_ed_stage1 += ed1
        self.sum_ed_stage2 += ed2
        self.sum_cer_stage1 += cer1
        self.sum_cer_stage2 += cer2

        em1 = int(s1 == ref)
        em2 = int(s2 == ref)
        self.sum_em_stage1 += em1
        self.sum_em_stage2 += em2

        if s1 != s2:
            self.changed += 1
            delta = cer2 - cer1
            if delta < -CFG["cer_eps"]:
                self.improved += 1
            elif delta > CFG["cer_eps"]:
                self.worsened += 1
            else:
                self.neutral_changed += 1

            if em1 == 0 and em2 == 1:
                self.fixed_em += 1
            if em1 == 1 and em2 == 0:
                self.broke_em += 1

    def finalize(self) -> Dict[str, Any]:
        micro1 = (self.sum_ed_stage1 / max(self.sum_ref_len, 1)) if self.sum_ref_len > 0 else None
        micro2 = (self.sum_ed_stage2 / max(self.sum_ref_len, 1)) if self.sum_ref_len > 0 else None

        macro1 = (self.sum_cer_stage1 / max(self.n, 1)) if self.n > 0 else None
        macro2 = (self.sum_cer_stage2 / max(self.n, 1)) if self.n > 0 else None

        em1 = (self.sum_em_stage1 / max(self.n, 1)) if self.n > 0 else None
        em2 = (self.sum_em_stage2 / max(self.n, 1)) if self.n > 0 else None

        correct_ratio = (self.improved / max(self.changed, 1)) if self.changed > 0 else None
        wrong_ratio = (self.worsened / max(self.changed, 1)) if self.changed > 0 else None

        return {
            "n": self.n,
            "cer_micro_stage1": micro1,
            "cer_micro_stage2": micro2,
            "cer_macro_stage1": macro1,
            "cer_macro_stage2": macro2,
            "exact_match_stage1": em1,
            "exact_match_stage2": em2,
            "changed": self.changed,
            "improved": self.improved,
            "worsened": self.worsened,
            "neutral_changed": self.neutral_changed,
            "fixed_em": self.fixed_em,
            "broke_em": self.broke_em,
            "correct_correction_ratio": correct_ratio,
            "wrong_correction_ratio": wrong_ratio,
        }


def main():
    ensure_dir(OUT_DIR)

    rows, csv_meta = load_eval_csv(IN_CSV)

    eval_cols = set([normalize_column_name(x) for x in CFG["eval_columns"]])
    char_names = load_character_names(LEXICON_PATH)
    has_char_lexicon = len(char_names) > 0

    agg_all = Agg()
    agg_by_col: Dict[str, Agg] = {}
    agg_by_error: Dict[str, Agg] = {}

    # character-focused counts (heuristic, substring-based)
    char_counts = {
        "lexicon_loaded": has_char_lexicon,
        "n_samples_with_ref_char": 0,
        "n_changed_samples_with_ref_char": 0,

        # sample-level: hits in GT increased/decreased
        "n_char_hit_increased_samples": 0,
        "n_char_hit_decreased_samples": 0,

        # event-level counts
        "n_correct_char_add_events": 0,
        "n_wrong_char_remove_events": 0,
        "n_spurious_char_add_events": 0,
    }

    # dialogue quote-focused counts
    quote_counts = {
        "n_dialogue_samples": 0,
        "n_dialogue_changed_samples": 0,
        "n_quote_ref_nonempty": 0,
        "n_quote_improved": 0,
        "n_quote_worsened": 0,
        "n_quote_neutral": 0,
    }

    # book-keeping
    skipped = {
        "missing_gt": 0,
        "column_not_eval": 0,
        "empty_stage2_treated_as_stage1": 0,
    }

    details: List[Dict[str, Any]] = []

    for r in rows:
        cut = parse_int(r.get("cut"))
        page = parse_int(r.get("page"))
        row = parse_int(r.get("row"))

        col_raw = r.get("column", "")
        col = normalize_column_name(col_raw)

        if col not in eval_cols:
            skipped["column_not_eval"] += 1
            continue

        s1_raw = r.get("stage1_ocr", "")
        s2_raw = r.get("stage2_corrected", "")
        gt_raw = r.get("final_gt", "")

        s1 = normalize_text(s1_raw)
        gt = normalize_text(gt_raw)

        if gt == "":
            skipped["missing_gt"] += 1
            continue

        if (s2_raw is None) or (str(s2_raw).strip() == ""):
            if CFG["treat_empty_stage2_as_stage1"]:
                s2 = s1
                skipped["empty_stage2_treated_as_stage1"] += 1
            else:
                s2 = normalize_text(s2_raw)
        else:
            s2 = normalize_text(s2_raw)

        cer1, ed1, ref_len = compute_cer(gt, s1)
        cer2, ed2, _ = compute_cer(gt, s2)

        # update aggregators
        agg_all.update(gt, s1, s2, cer1, cer2, ed1, ed2, ref_len)

        if col not in agg_by_col:
            agg_by_col[col] = Agg()
        agg_by_col[col].update(gt, s1, s2, cer1, cer2, ed1, ed2, ref_len)

        err = (r.get("error_type", "") or "").strip()
        if err == "":
            err = "unknown"
        if err not in agg_by_error:
            agg_by_error[err] = Agg()
        agg_by_error[err].update(gt, s1, s2, cer1, cer2, ed1, ed2, ref_len)

        changed = int(s1 != s2)
        delta = cer2 - cer1

        outcome = "unchanged"
        if changed:
            if delta < -CFG["cer_eps"]:
                outcome = "improved"
            elif delta > CFG["cer_eps"]:
                outcome = "worsened"
            else:
                outcome = "neutral"

        em1 = int(s1 == gt)
        em2 = int(s2 == gt)

        diff = compute_diff_span(s1, s2)

        # ---- character name analysis (optional) ----
        ref_char_names: List[str] = []
        char_hits_before = char_hits_after = 0
        char_hit_delta = 0
        spurious_add = 0
        wrong_remove = 0
        correct_add = 0

        if has_char_lexicon:
            # which lexicon names appear in GT
            for n in char_names:
                if n and (n in gt):
                    ref_char_names.append(n)

            if ref_char_names:
                char_counts["n_samples_with_ref_char"] += 1
                if changed:
                    char_counts["n_changed_samples_with_ref_char"] += 1

                char_hits_before = sum(1 for n in ref_char_names if n in s1)
                char_hits_after = sum(1 for n in ref_char_names if n in s2)
                char_hit_delta = char_hits_after - char_hits_before

                if char_hit_delta > 0:
                    char_counts["n_char_hit_increased_samples"] += 1
                elif char_hit_delta < 0:
                    char_counts["n_char_hit_decreased_samples"] += 1

                # event-level: correct add / wrong remove for GT names
                for n in ref_char_names:
                    b_has = (n in s1)
                    a_has = (n in s2)
                    if (not b_has) and a_has:
                        correct_add += 1
                    if b_has and (not a_has):
                        wrong_remove += 1

                char_counts["n_correct_char_add_events"] += correct_add
                char_counts["n_wrong_char_remove_events"] += wrong_remove

            # spurious additions: names appear in s2 but not in gt and not in s1
            # (heuristic: counts per sample, summed)
            for n in char_names:
                if len(n) < 2:
                    continue
                if (n in s2) and (n not in s1) and (n not in gt):
                    spurious_add += 1
            char_counts["n_spurious_char_add_events"] += spurious_add

        # ---- dialogue quote analysis ----
        quote_ref = quote_s1 = quote_s2 = ""
        quote_cer1 = quote_cer2 = None
        quote_delta = None
        quote_outcome = "n/a"

        if col == "dialogue":
            quote_counts["n_dialogue_samples"] += 1
            if changed:
                quote_counts["n_dialogue_changed_samples"] += 1

            quote_ref = extract_quotes(gt)
            quote_s1 = extract_quotes(s1)
            quote_s2 = extract_quotes(s2)

            if quote_ref != "":
                quote_counts["n_quote_ref_nonempty"] += 1
                quote_cer1, _, _ = compute_cer(quote_ref, quote_s1)
                quote_cer2, _, _ = compute_cer(quote_ref, quote_s2)
                quote_delta = quote_cer2 - quote_cer1

                if quote_delta < -CFG["cer_eps"]:
                    quote_counts["n_quote_improved"] += 1
                    quote_outcome = "quote_improved"
                elif quote_delta > CFG["cer_eps"]:
                    quote_counts["n_quote_worsened"] += 1
                    quote_outcome = "quote_worsened"
                else:
                    quote_counts["n_quote_neutral"] += 1
                    quote_outcome = "quote_neutral"
            else:
                quote_outcome = "no_quote_in_ref"

        details.append({
            "cut": cut,
            "page": page,
            "row": row,
            "column": col,
            "error_type": err,
            "note": clip_text(str(r.get("note", "") or ""), 200),

            "ref": clip_text(gt, CFG["max_text_len_in_details"]),
            "stage1": clip_text(s1, CFG["max_text_len_in_details"]),
            "stage2": clip_text(s2, CFG["max_text_len_in_details"]),

            "changed": changed,
            "cer_stage1": cer1,
            "cer_stage2": cer2,
            "cer_delta": delta,
            "em_stage1": em1,
            "em_stage2": em2,
            "outcome": outcome,
            "fixed_em": int(em1 == 0 and em2 == 1),
            "broke_em": int(em1 == 1 and em2 == 0),

            "diff_from": clip_text(diff.get("from_sub", ""), 250) if diff.get("changed") else "",
            "diff_to": clip_text(diff.get("to_sub", ""), 250) if diff.get("changed") else "",

            "ref_char_names": "|".join(ref_char_names) if ref_char_names else "",
            "char_hits_stage1": char_hits_before,
            "char_hits_stage2": char_hits_after,
            "char_hit_delta": char_hit_delta,
            "char_correct_add_events": correct_add,
            "char_wrong_remove_events": wrong_remove,
            "char_spurious_add_events": spurious_add,

            "quote_ref": clip_text(quote_ref, 250),
            "quote_stage1": clip_text(quote_s1, 250),
            "quote_stage2": clip_text(quote_s2, 250),
            "quote_cer_stage1": quote_cer1,
            "quote_cer_stage2": quote_cer2,
            "quote_cer_delta": quote_delta,
            "quote_outcome": quote_outcome,
        })

    # finalize summaries
    by_col = {k: v.finalize() for k, v in agg_by_col.items()}
    by_err = {k: v.finalize() for k, v in agg_by_error.items()}

    summary = {
        "meta": {
            "created_at": now_iso(),
            "episode_id": EPISODE_ID,
            "stage2_variant": STAGE2_VARIANT,
            "in_csv": IN_CSV,
            "csv_meta": csv_meta,
            "lexicon_path": LEXICON_PATH if os.path.isfile(LEXICON_PATH) else None,
            "eval_columns": sorted(list(eval_cols)),
            "normalization": {
                "neutral_symbols_removed": CFG["neutral_symbols_removed"],
                "collapse_spaces": CFG["collapse_spaces"],
                "keep_newlines": CFG["keep_newlines"],
                "treat_empty_stage2_as_stage1": CFG["treat_empty_stage2_as_stage1"],
            }
        },

        # requested items
        "requested_metrics": {
            "修正した文数": agg_all.changed,  # raw != corrected
            "補正の失敗数_CER悪化": agg_all.worsened,
            "補正の失敗数_EM破壊": agg_all.broke_em,
            "正しい補正割合": (agg_all.improved / agg_all.changed) if agg_all.changed > 0 else None,
            "間違った補正割合": (agg_all.worsened / agg_all.changed) if agg_all.changed > 0 else None,
        },

        # extra helpful summaries
        "text_quality_total": agg_all.finalize(),
        "text_quality_by_column": by_col,
        "text_quality_by_error_type": by_err,

        "character_name_analysis": char_counts,
        "dialogue_quote_analysis": quote_counts,
        "skipped": skipped,

        "notes": [
            "『正しいキャラ名の補正数』は、GT内に登場する辞書キャラ名が stage1→stage2 で増えた（ヒット増）/ イベント（correct_add_events）で近似しています。",
            "辞書に無い略称・誤字のファジー一致は行っていないため、キャラ名補正の真の成功を過小評価する可能性があります（ただし誤爆の検出には強い）。",
            "『正しいセリフの補正数』は dialogue 列に対して、カギ括弧「」内のCERが改善した件数で別途集計しています。",
            "失敗は2通り出します: (1) CERが悪化した (2) もともと完全一致だったのに壊した（broke_em）。",
        ]
    }

    ensure_dir(OUT_DIR)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {OUT_JSON}")

    # write details CSV
    with open(OUT_DETAILS_CSV, "w", encoding="utf-8", newline="") as f:
        if details:
            writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
            writer.writeheader()
            for d in details:
                writer.writerow(d)
    print(f"[OK] wrote {OUT_DETAILS_CSV}")

    # quick print
    print("\n=== QUICK SUMMARY ===")
    print("total evaluated n:", agg_all.n)
    print("changed:", agg_all.changed, "improved:", agg_all.improved, "worsened:", agg_all.worsened, "neutral_changed:", agg_all.neutral_changed)
    print("fixed_em:", agg_all.fixed_em, "broke_em:", agg_all.broke_em)
    print("missing_gt:", skipped["missing_gt"], "column_not_eval:", skipped["column_not_eval"])


if __name__ == "__main__":
    main()

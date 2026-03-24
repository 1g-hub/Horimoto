# stage2_align_and_clean.py

import os
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

# ==================================================
# GLOBAL PARAMS (EDIT HERE)
# ==================================================
APP_ROOT = "/app"  # Docker: -v $(PWD):/app

EPISODE_ID = "episode01"

# Stage1 outputs
STAGE1_OUT_ROOT = f"outputs/{EPISODE_ID}"  # 例: /app/outputs/episode01
STAGE1_INDEX = f"{STAGE1_OUT_ROOT}/index.stage1.json"

# Stage0 outputs (script knowledge base)
SCRIPT_PHASE = "script_phase1"
STAGE0_OUT_ROOT = f"outputs/{SCRIPT_PHASE}/{EPISODE_ID}"
LEXICON_PATH = f"{STAGE0_OUT_ROOT}/lexicon_entities.json"
SCRIPT_CHUNKS_JSONL = f"{STAGE0_OUT_ROOT}/script_chunks.jsonl"

# Stage2 outputs
STAGE2_OUT_ROOT = f"outputs_stage2/{EPISODE_ID}"
STAGE2_CUTS_DIR = f"{STAGE2_OUT_ROOT}/cuts"
STAGE2_INDEX = f"{STAGE2_OUT_ROOT}/index.stage2.json"

CFG = {
    # Process range: "all" or "range"
    "cut_select": "all",           # "all" | "range"
    "cut_start": 1,
    "cut_end": 9999,

    # Which OCR columns to use (if missing, handled gracefully)
    "use_columns": ["action_memo", "dialogue", "time"],

    # Script candidate retrieval
    "script_top_k": 5,
    "script_min_score": 0.80,

    # Entity extraction
    "max_entities": 20,

    # Basic normalization
    "normalize_spaces": True,

    # Name correction
    "name_correction_min_sim": 0.66,  # 0.8くらいが無難
}
# ==================================================


# -------------------------
# Regex / helpers
# -------------------------
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
SE_RE = re.compile(r"(?:^|\b)(SE|ＳＥ)\s*[:：]?\s*(.+)$")
MA_RE = re.compile(r"(?:^|\b)(MA|ＭＡ|BGM|ＢＧＭ)\s*[:：]?\s*(.+)$")
OL_RE = re.compile(r"(O\.?L\.?|Ｏ\.?Ｌ\.?)", re.IGNORECASE)

PAREN_RE = re.compile(r"（[^）]*）")
QUOTE_ONLY_RE = re.compile(r"「([^」]+)」")


def collapse_placeholder_lines(text: str, placeholder: str = "□") -> str:
    if not text:
        return text

    out_lines = []
    prev_was_placeholder = False

    for ln in text.splitlines():
        t = ln.strip()
        is_placeholder_only = (t == placeholder)

        if is_placeholder_only:
            if not prev_was_placeholder:
                out_lines.append(placeholder)
            prev_was_placeholder = True
        else:
            out_lines.append(ln)
            prev_was_placeholder = False

    return "\n".join(out_lines)


def extract_quote_text_for_query(dialogue_raw: str) -> str:
    quotes = QUOTE_ONLY_RE.findall(dialogue_raw or "")
    quotes = [q.strip() for q in quotes if q.strip() and q.strip() != "□"]
    if quotes:
        return "\n".join(quotes)

    lines = []
    for ln in (dialogue_raw or "").splitlines():
        t = ln.strip()
        if not t or t == "□":
            continue
        if t.startswith("S：") or t.startswith("S:"):
            continue
        if re.fullmatch(r"\(.*\)", t):
            continue
        lines.append(t)
    return "\n".join(lines)


def guess_speaker_hint(dialogue_raw: str) -> Optional[str]:
    for ln in (dialogue_raw or "").splitlines():
        t = ln.strip()
        if not t or t == "□":
            continue
        if t.endswith("Ｍ") or t.endswith("M"):
            return t.replace("Ｍ", "M")
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if CFG["normalize_spaces"]:
        s = s.replace("\u3000", " ")
        s = re.sub(r"\s+", " ", s).strip()
    return s


def join_lines_from_stage1_cell(cell: Dict[str, Any]) -> str:
    if not cell:
        return ""
    raw = cell.get("raw_text")
    if raw and isinstance(raw, str):
        return raw.strip()
    lines = cell.get("lines")
    if isinstance(lines, list):
        return "\n".join([str(x) for x in lines]).strip()
    return ""


# -------------------------
# Script KB loading
# -------------------------
@dataclass
class ScriptChunk:
    chunk_id: str
    paragraph_index: int
    text: str
    kind: str
    scene_id: Optional[str]
    speaker: Optional[str]
    speaker_note: Optional[str]


def load_script_chunks(path_jsonl: str) -> List[ScriptChunk]:
    chunks: List[ScriptChunk] = []
    for obj in iter_jsonl(path_jsonl):
        chunks.append(
            ScriptChunk(
                chunk_id=obj.get("chunk_id", ""),
                paragraph_index=int(obj.get("paragraph_index", -1)),
                text=str(obj.get("text", "")),
                kind=str(obj.get("kind", "")),
                scene_id=obj.get("scene_id"),
                speaker=obj.get("speaker"),
                speaker_note=obj.get("speaker_note"),
            )
        )
    return chunks


def load_lexicon_characters(path_json: str) -> List[str]:
    obj = read_json(path_json)
    chars = []
    for e in obj.get("entities", []):
        if e.get("type") == "character":
            name = e.get("canonical")
            if isinstance(name, str) and name.strip():
                chars.append(name.strip())
    chars.sort(key=lambda x: (-len(x), x))
    return chars


def build_name_variants(characters: List[str]) -> List[str]:
    """
    フルネーム + 中点先頭名（ベリル・ガーデナント -> ベリル）を候補にする
    """
    out = []
    seen = set()
    for name in characters:
        if name not in seen:
            seen.add(name)
            out.append(name)
        if "・" in name:
            short = name.split("・")[0]
            if short and short not in seen:
                seen.add(short)
                out.append(short)
    out.sort(key=lambda x: (-len(x), x))
    return out

def build_speaker_variants_from_script(script_chunks: List[ScriptChunk]) -> List[str]:
    """
    script_chunks の speaker から候補を作る（正解側の表記）。
    - "ベリルM" -> "ベリルM", "ベリル"
    - "ベリル・ガーデナント" -> "ベリル・ガーデナント", "ベリル"
    """
    out = []
    seen = set()

    for ch in script_chunks:
        sp = ch.speaker
        if not sp or not isinstance(sp, str):
            continue
        s = sp.strip().replace("Ｍ", "M")
        if not s:
            continue

        # そのまま
        if s not in seen:
            seen.add(s); out.append(s)

        # 末尾Mを外した版
        if s.endswith("M"):
            b = s[:-1]
            if b and b not in seen:
                seen.add(b); out.append(b)

        # 中点の先頭
        if "・" in s:
            head = s.split("・")[0]
            if head and head not in seen:
                seen.add(head); out.append(head)

    out.sort(key=lambda x: (-len(x), x))
    return out



# -------------------------
# Parsing / tagging
# -------------------------
def extract_time_tokens(text: str) -> List[str]:
    return TIME_RE.findall(text or "")


def parse_audio_and_directions(lines: List[str]) -> Dict[str, Any]:
    se = []
    ma = []
    directions = []
    ol = []
    other_lines = []

    for ln in lines:
        t = ln.strip()
        if not t or t == "□":
            continue

        m = SE_RE.search(t)
        if m:
            se.append(norm_text(m.group(2)))
            continue

        m = MA_RE.search(t)
        if m:
            ma.append(norm_text(m.group(2)))
            continue

        if OL_RE.search(t):
            ol.append(t)

        ps = PAREN_RE.findall(t)
        if ps:
            directions.extend(ps)
            continue

        other_lines.append(t)

    return {
        "se": se,
        "ma": ma,
        "directions": directions,
        "ol": ol,
        "other_lines": other_lines,
    }


def find_characters_in_text(text: str, characters: List[str]) -> List[str]:
    found = []
    seen = set()
    for name in characters:
        if name in text and name not in seen:
            seen.add(name)
            found.append(name)
            if len(found) >= CFG["max_entities"]:
                break
    return found


# -------------------------
# Script candidate retrieval (simple)
# -------------------------
def simple_overlap_score(query: str, text: str) -> float:
    q = norm_text(query)
    t = norm_text(text)
    if not q or not t:
        return 0.0

    def bigrams(s: str) -> set:
        s = s.replace(" ", "")
        if len(s) < 2:
            return set()
        return {s[i:i+2] for i in range(len(s)-1)}

    qb = bigrams(q)
    tb = bigrams(t)
    if not qb or not tb:
        return 0.0
    inter = len(qb & tb)
    return inter / max(len(qb), 1)


def retrieve_script_candidates(query: str, script_chunks: List[ScriptChunk], speaker_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    scored = []
    for ch in script_chunks:
        score = simple_overlap_score(query, ch.text)
        if score <= 0:
            continue

        bonus = 0.0
        if speaker_hint and ch.speaker:
            sh = norm_text(speaker_hint).replace("Ｍ", "M")
            sp = norm_text(ch.speaker).replace("Ｍ", "M")
            if sh == sp:
                bonus = 0.12
            else:
                if sh.endswith("M") and sh[:-1] == sp:
                    bonus = 0.06

        scored.append((score + bonus, score, ch))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    out = []
    for score2, score_raw, ch in scored[: CFG["script_top_k"]]:
        if score2 < CFG["script_min_score"]:
            continue
        out.append({
            "chunk_id": ch.chunk_id,
            "score": round(float(score2), 4),
            "score_raw": round(float(score_raw), 4),
            "paragraph_index": ch.paragraph_index,
            "kind": ch.kind,
            "scene_id": ch.scene_id,
            "speaker": ch.speaker,
            "speaker_note": ch.speaker_note,
            "text": ch.text,
        })
    return out


# -------------------------
# Name correction (Stage2+)
# -------------------------
def normalize_speaker_token(tok: str) -> str:
    t = norm_text(tok)
    t = t.replace("Ｍ", "M")
    t = t.replace(" ", "")
    return t


def levenshtein(a: str, b: str) -> int:
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
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def adjacent_transposition_distance(a: str, b: str) -> int:
    """
    a と b が「隣接2文字の入れ替え」1回で一致するなら 1 を返す。
    それ以外は大きい値を返す。
    例: ベルリ <-> ベリル は 1
    """
    if len(a) != len(b):
        return 10**9
    if a == b:
        return 0

    n = len(a)
    for i in range(n - 1):
        if (
            a[:i] == b[:i]
            and a[i] == b[i + 1]
            and a[i + 1] == b[i]
            and a[i + 2 :] == b[i + 2 :]
        ):
            return 1
    return 10**9


def similarity(a: str, b: str) -> float:
    d = levenshtein(a, b)
    m = max(len(a), len(b), 1)
    return 1.0 - (d / m)


def correct_speaker_lines_in_dialogue(
    dialogue_raw: str,
    name_variants: List[str],
    min_sim: float
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    return: (dialogue_norm, corrections, debug_rows)
    比較では候補側の末尾Mも除去して距離を計算する。
    """
    corrections: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    lines = (dialogue_raw or "").splitlines()
    out_lines: List[str] = []

    for ln in lines:
        t0 = ln.strip()
        if not t0 or t0 == "□":
            out_lines.append(ln)
            continue

        # 台詞本文は触らない
        if t0.startswith("「") and t0.endswith("」"):
            out_lines.append(ln)
            continue

        # ノイズ行は除外
        if t0.startswith("S：") or t0.startswith("S:"):
            out_lines.append(ln)
            continue
        if re.fullmatch(r"\(.*\)", t0):
            out_lines.append(ln)
            continue

        norm_tok = normalize_speaker_token(t0)

        # OCR側 suffix
        suffix = ""
        base = norm_tok
        if base.endswith("M"):
            suffix = "M"
            base = base[:-1]

        best = None
        best_sim = -1.0
        best_dist = 10**9
        best_cmp = None  # 比較に使った候補文字列

        for cand in name_variants:
            cand_norm = normalize_speaker_token(cand)

            # ★候補側も比較では末尾Mを外す
            cand_cmp = cand_norm[:-1] if cand_norm.endswith("M") else cand_norm

            d_lev = levenshtein(base, cand_cmp)
            d_tr  = adjacent_transposition_distance(base, cand_cmp)
            d = min(d_lev, d_tr)
            sim = 1.0 - d / max(len(base), len(cand_cmp), 1)

            if sim > best_sim or (sim == best_sim and d < best_dist):
                best_sim = sim
                best_dist = d
                best = cand
                best_cmp = cand_cmp

        # allow判定：通常閾値 or 短い名前なら距離1でOK
        allow = False
        if best is not None and best_sim >= min_sim:
            allow = True
        if not allow and best is not None and len(base) <= 5 and best_dist <= 1:
            allow = True

        # fixed生成：bestがM付きでも二重にしない
        fixed = t0
        if allow and best is not None:
            fixed_base = best
            fixed_base_norm = normalize_speaker_token(fixed_base)
            if fixed_base_norm.endswith("M"):
                fixed_base = fixed_base[:-1]
            fixed = fixed_base + suffix

        debug_rows.append({
            "line": t0,
            "base": base,
            "suffix": suffix,
            "best": best,
            "best_cmp": best_cmp,
            "best_sim": round(best_sim, 4) if best is not None else None,
            "best_dist": int(best_dist) if best is not None else None,
            "min_sim": min_sim,
            "allow": allow,
            "fixed": fixed,
            "name_variants_size": len(name_variants),
        })

        if allow and best is not None and fixed != t0:
            corrections.append({
                "field": "ocr_raw.dialogue",
                "from": t0,
                "to": fixed,
                "rule": "lexicon+script_speaker_near_match",
                "confidence": round(best_sim, 4),
                "distance": int(best_dist),
                "best_candidate": best,
                "best_candidate_cmp": best_cmp,
            })
            out_lines.append(fixed)
        else:
            out_lines.append(ln)

    return "\n".join(out_lines), corrections, debug_rows

# -------------------------
# Stage1 cut reading
# -------------------------
def list_stage1_cuts(index_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    cuts = index_obj.get("cuts", [])
    out = []
    for c in cuts:
        cut_num = c.get("cut")
        out_json = c.get("out_json")
        if out_json is None:
            continue
        out.append({"cut": cut_num, "out_json": out_json})
    return out


def cut_in_range(cut_num: Optional[int]) -> bool:
    if CFG["cut_select"] == "all":
        return True
    if cut_num is None:
        return False
    return CFG["cut_start"] <= int(cut_num) <= CFG["cut_end"]


# -------------------------
# Main Stage2
# -------------------------
def stage2_process_one_cut(
    stage1_cut_path: str,
    characters: List[str],
    name_variants: List[str],
    script_chunks: List[ScriptChunk]
) -> Dict[str, Any]:
    obj = read_json(stage1_cut_path)

    cut_num = obj.get("cut")
    span = obj.get("span")
    rows = obj.get("rows", [])

    col_texts_raw: Dict[str, List[str]] = {c: [] for c in CFG["use_columns"]}
    time_tokens: List[str] = []

    for r in rows:
        cols = r.get("cols", {})
        for col in CFG["use_columns"]:
            cell = cols.get(col, {})
            txt = join_lines_from_stage1_cell(cell)
            if txt:
                col_texts_raw[col].append(txt)

    action_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("action_memo", [])))
    dialogue_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("dialogue", [])))
    time_raw = collapse_placeholder_lines("\n".join(col_texts_raw.get("time", [])))

    # Stage2+ name correction (speaker lines in dialogue)
    dialogue_norm, name_corrections, name_debug = correct_speaker_lines_in_dialogue(
        dialogue_raw,
        name_variants=name_variants,
        min_sim=CFG["name_correction_min_sim"],
    )


    for src in [time_raw, action_raw, dialogue_raw]:
        time_tokens.extend(extract_time_tokens(src))
    seen = set()
    time_tokens_unique = []
    for t in time_tokens:
        if t not in seen:
            seen.add(t)
            time_tokens_unique.append(t)

    dialogue_lines = [ln for ln in dialogue_raw.splitlines() if ln.strip()]
    action_lines = [ln for ln in action_raw.splitlines() if ln.strip()]

    parsed_dialogue = parse_audio_and_directions(dialogue_lines)
    parsed_action = parse_audio_and_directions(action_lines)

    merged_text_for_entities = "\n".join([action_raw, dialogue_norm])
    chars_found = find_characters_in_text(merged_text_for_entities, characters)

    speaker_hint = guess_speaker_hint(dialogue_raw)

    query = extract_quote_text_for_query(dialogue_raw).strip()
    if not query:
        query = action_raw.strip()

    script_candidates = retrieve_script_candidates(query, script_chunks, speaker_hint=speaker_hint) if query else []

    out = {
        "cut": cut_num,
        "span": span,
        "source_stage1": stage1_cut_path,

        "ocr_raw": {
            "action_memo": action_raw,
            "dialogue": dialogue_raw,
            "time_raw": time_raw,
            "time_tokens": time_tokens_unique,
        },

        # Stage2+: normalize/corrected text (rawは保持)
        "ocr_norm": {
            "dialogue": dialogue_norm,
        },

        "parsed": {
            "dialogue": parsed_dialogue,
            "action_memo": parsed_action,
        },

        "entities": {
            "characters": chars_found,
        },

        "script_candidates": script_candidates,

        "corrections": name_corrections,

        "meta": {
            "created_at": now_iso(),
            "stage2_version": "0.2-name-correction",
        }, 

        "debug": {
            "name_correction": name_debug[:20]  # まずは先頭20件だけ
        },
    }
    return out


def main():
    for p in [STAGE1_INDEX, LEXICON_PATH, SCRIPT_CHUNKS_JSONL]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    ensure_dir(STAGE2_OUT_ROOT)
    ensure_dir(STAGE2_CUTS_DIR)

    stage1_index = read_json(STAGE1_INDEX)
    stage1_cut_refs = list_stage1_cuts(stage1_index)

    characters = load_lexicon_characters(LEXICON_PATH)
    script_chunks = load_script_chunks(SCRIPT_CHUNKS_JSONL)

    name_variants = build_name_variants(characters)
    name_variants_script = build_speaker_variants_from_script(script_chunks)

    # merge unique
    seen = set(name_variants)
    for s in name_variants_script:
        if s not in seen:
            seen.add(s)
            name_variants.append(s)

    # 長い順
    name_variants.sort(key=lambda x: (-len(x), x))

    index_out = {
        "episode_id": EPISODE_ID,
        "source": {
            "stage1_index": STAGE1_INDEX,
            "lexicon": LEXICON_PATH,
            "script_chunks": SCRIPT_CHUNKS_JSONL,
        },
        "cuts": [],
        "meta": {
            "created_at": now_iso(),
            "stage2_version": "0.2-name-correction",
            "cfg": CFG,
            "stats": {
                "lexicon_characters": len(characters),
                "name_variants": len(name_variants),
                "script_chunks": len(script_chunks),
            }
        }
    }

    for ref in stage1_cut_refs:
        cut_num = ref.get("cut")
        if not cut_in_range(cut_num):
            continue

        stage1_cut_path = ref["out_json"]
        if not os.path.isabs(stage1_cut_path):
            stage1_cut_path = os.path.join(APP_ROOT, stage1_cut_path)

        if not os.path.isfile(stage1_cut_path):
            index_out["cuts"].append({
                "cut": cut_num,
                "stage1_path": stage1_cut_path,
                "stage2_path": None,
                "status": "missing_stage1_file",
            })
            continue

        stage2_obj = stage2_process_one_cut(stage1_cut_path, characters, name_variants, script_chunks)

        if cut_num is None:
            out_name = f"cut_unknown_{len(index_out['cuts']):06d}.stage2.json"
        else:
            out_name = f"cut{int(cut_num):04d}.stage2.json"
        out_path = os.path.join(STAGE2_CUTS_DIR, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stage2_obj, f, ensure_ascii=False, indent=2)

        index_out["cuts"].append({
            "cut": cut_num,
            "stage1_path": stage1_cut_path,
            "stage2_path": out_path,
            "status": "ok",
        })

        print(f"[OK] wrote {out_path}")

    with open(STAGE2_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {STAGE2_INDEX}")


if __name__ == "__main__":
    main()

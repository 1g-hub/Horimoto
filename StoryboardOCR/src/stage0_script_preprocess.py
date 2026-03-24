# stage0_script_preprocess.py

import os
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set

from docx import Document

# =========================
# GLOBAL PARAMS (EDIT HERE)
# =========================
PHASE = "script_phase1"
EPISODE_ID = "episode01"
DATA_ROOT = "data"
OUTPUT_ROOT = "outputs"
# =========================

# ==================================================
# CONFIG
# ==================================================
CFG = {
    # 入力脚本（.docx）
    "script_docx": f"{DATA_ROOT}/{PHASE}/{EPISODE_ID}.docx",

    # 出力先ベースディレクトリ
    "out_dir": f"{OUTPUT_ROOT}/{PHASE}/{EPISODE_ID}",

    # エピソードID（Stage1 / Stage2 と揃える）
    "episode_id": EPISODE_ID,

    # 出力ファイル名
    "lexicon_filename": "lexicon_entities.json",
    "chunks_filename": "script_chunks.jsonl",

    # ヒューリスティック設定
    # "min_katakana_len": 3,
    "max_evidence_per_entity": 10,

    # 地の文の分割：句点で1文ずつ chunk にする
    "split_narration_by_kuten": True,

    # セリフの分割：句点で1文ずつ chunk にする
    "split_dialogue_by_punct": False,

    # 話者行の最大長（辞書一致があっても長すぎは弾く）
    "max_speaker_label_len": 12,
}
# ==================================================


# -------------------------
# Patterns / Rules
# -------------------------
HEADING_CHAR_SECTION = "【登場人物】"
CHAR_SECTION_END_TOKEN = "他"

# 丸数字のシーン見出し（①②③... / ⑩など）
# ①-⑳(U+2460..U+2473), ㉑-㉟(U+3251..U+325F), ㊱-㊿(U+32B1..U+32BF)
CIRCLED_NUM_RE = re.compile(r"^[①-⑳㉑-㉟㊱-㊿]")

# 注記（※）
NOTE_RE = re.compile(r"^※")

# 1段落内に複数「」があることを想定して抽出
QUOTE_RE = re.compile(r"「([^」]*)」")

# "話者「...」" の話者部分を拾う（鉤括弧の前にあるラベル）
SPEAKER_BEFORE_QUOTE_RE = re.compile(r"^(?P<speaker>[^「」]{1,30})「")


# 空白正規化
def norm(s: str) -> str:
    s = (s or "").replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)

    # 点々の正規化
    # 1) 三点リーダ…の連続を1つに
    s = re.sub(r"…{2,}", "…", s)
    # 2) 中点・の連続（・・・）を三点リーダに寄せる（必要なら）
    s = re.sub(r"・{3,}", "…", s)

    return s

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_snippet(s: str, n: int = 80) -> str:
    s = norm(s)
    return s[:n] + ("…" if len(s) > n else "")

def is_time_like(s: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\b", s))


# -------------------------
# DOCX reading
# -------------------------
def read_docx_paragraphs(docx_path: str) -> List[Dict[str, Any]]:
    doc = Document(docx_path)
    paras = []
    for i, p in enumerate(doc.paragraphs):
        text = norm(p.text)
        if not text:
            continue
        paras.append({"paragraph_index": i, "text": text})
    return paras


# -------------------------
# Stage0: entity lexicon from 【登場人物】 ... 他
# -------------------------
def build_lexicon_entities(paras: List[Dict[str, Any]]) -> Dict[str, Any]:
    in_char = False
    entities: Dict[str, Dict[str, Any]] = {}

    for p in paras:
        t = p["text"]

        if t == HEADING_CHAR_SECTION:
            in_char = True
            continue

        if in_char:
            if t == CHAR_SECTION_END_TOKEN:
                in_char = False
                continue

            canonical = t
            if canonical and canonical not in entities:
                entities[canonical] = {
                    "canonical": canonical,
                    "type": "character",
                    "aliases": [],
                    "evidence": [],
                }
            if canonical:
                entities[canonical]["evidence"].append({
                    "paragraph_index": p["paragraph_index"],
                    "text_snippet": safe_snippet(t),
                })

    # evidence を間引く
    for ent in entities.values():
        ent["evidence"] = ent["evidence"][: CFG["max_evidence_per_entity"]]

    entity_list = sorted(entities.values(), key=lambda x: (-len(x["canonical"]), x["canonical"]))

    return {
        "created_at": now_iso(),
        "episode_id": CFG["episode_id"],
        "source_docx": CFG["script_docx"],
        "entities": entity_list,
        "stats": {"entity_count": len(entity_list)},
    }


def build_speaker_set_from_lexicon(lexicon: Dict[str, Any]) -> set[str]:
    """
    speaker候補を作る：
    - canonical
    - 中点名の先頭（ベリル・ガーデナント -> ベリル）
    """
    s = set()
    for e in lexicon.get("entities", []):
        if e.get("type") != "character":
            continue
        name = e.get("canonical", "")
        if name:
            s.add(name)
            if "・" in name:
                s.add(name.split("・")[0])
    return {x for x in s if x}


def normalize_speaker_label(label: str) -> str:
    t = norm(label)
    # 全角Mを半角に
    t = t.replace("Ｍ", "M")
    return t


def is_speaker_line(text: str, speaker_set: set[str]) -> bool:
    """
    地の文を誤って speaker_line にしないための厳しめ判定。
    """
    t = normalize_speaker_label(text)
    if not t:
        return False
    if len(t) > CFG["max_speaker_label_len"]:
        return False
    # 記号・句読点・括弧などを含むなら話者ではない
    if re.search(r"[。、，,.（）()～\-—/「」]", t):
        return False
    if is_time_like(t):
        return False
    # ベリルM みたいなモノローグ表記を許す
    if t.endswith("M") and t[:-1] in speaker_set:
        return True
    return t in speaker_set


# -------------------------
# Text splitting helpers
# -------------------------
def split_narration_sentences(text: str) -> List[str]:
    """
    地の文：句点「。」で分割（句点を残す）
    """
    text = norm(text)
    if not text:
        return []
    if not CFG["split_narration_by_kuten"]:
        return [text]

    out = []
    buf = ""
    for ch in text:
        buf += ch
        if ch == "。":
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [x for x in out if x]


def split_dialogue_sentences(quote_text: str) -> List[str]:
    """
    入力: '「...」' 形式
    出力: 句点/！？/… で分割した '「...」' を返す（句点等が無ければ1つ）
    - 「？」や「……」で終わる場合も、鉤括弧が閉じていればそれはセリフとして保持される
    """
    s = norm(quote_text)
    if not s:
        return []

    inner = s
    if inner.startswith("「"):
        inner = inner[1:]
    if inner.endswith("」"):
        inner = inner[:-1]

    if not CFG.get("split_dialogue_by_punct", True):
        return [f"「{inner}」"] if inner else []

    # 分割終端として扱う文字
    # 句点/疑問符/感嘆符/三点リーダ（…）を優先
    END_CHARS = {"。", "？", "！"}
    out = []
    buf = ""
    for ch in inner:
        buf += ch
        if ch in END_CHARS:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())

    return [f"「{p}」" for p in out if p]


def split_by_quotes(text: str) -> List[Dict[str, str]]:
    """
    1つの文字列から「…」を全抽出し、
    quote外→narration, quote→dialogue でセグメント化。
    """
    out: List[Dict[str, str]] = []
    last = 0
    for m in QUOTE_RE.finditer(text):
        if m.start() > last:
            n = norm(text[last:m.start()])
            if n:
                out.append({"kind": "narration", "text": n})
        out.append({"kind": "dialogue", "text": f"「{m.group(1)}」"})
        last = m.end()
    tail = norm(text[last:])
    if tail:
        out.append({"kind": "narration", "text": tail})
    return out


# 例: モルデア（オフ） / ベリル（背）
PAREN_NOTE_RE = re.compile(r"^(?P<name>.*?)(?:（(?P<note>[^）]*)）)?\s*$")

def try_extract_inline_speaker_prefix(text_with_quote: str, speaker_set: set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    返り値: (speaker, speaker_note)
      - speaker_note には "オフ" / "背" など括弧内を入れる
    """
    s = norm(text_with_quote)
    i = s.find("「")
    if i == -1:
        return None, None

    prefix = norm(s[:i])
    if not prefix:
        return None, None

    prefix = re.sub(r"[：:]\s*$", "", prefix).strip()

    m = PAREN_NOTE_RE.match(prefix)
    speaker_raw = m.group("name") if m else prefix
    note = m.group("note") if m else None
    note = norm(note) if note else None

    speaker = normalize_speaker_label(speaker_raw)
    if not speaker:
        return None, None

    # 辞書があれば参照（でも無くても採用する方針）
    if speaker_set:
        if speaker in speaker_set:
            return speaker, note
        if speaker.endswith("M") and speaker[:-1] in speaker_set:
            return speaker, note
        if "・" in speaker and speaker.split("・")[0] in speaker_set:
            return speaker, note

    return speaker, note


# -------------------------
# Stage0: chunking story
# -------------------------
def build_script_chunks(paras: List[Dict[str, Any]], speaker_set: set[str]) -> List[Dict[str, Any]]:
    """
    改善点：
      - 段落跨ぎのセリフ（「が開いて次段落で閉じる）を quote_buffer で正しく処理
      - 1段落に複数の「」があっても順序通りに dialogue/narration を生成
      - narration は段落跨ぎで連結し、句点で切る
      - セリフが始まる（「）直前に、句点の無い narration を強制確定
      - 「の直前prefixが話者ラベルなら narration に混ぜず speaker として消費
      - speaker_note（オフ/背など括弧内）を dialogue chunk に保存
      - NEW:
          * 《ＯＰ》/《ED》などを marker chunk として分離
          * × × × などの区切りを separator chunk として分離
    """
    # --- special patterns (inside function for copy-paste convenience) ---
    # 《ＯＰ》や《ED》
    MARKER_RE = re.compile(r"^(?P<tag>《[^》]+》)\s*(?P<rest>.*)$")
    # × が3個以上（間に空白OK）
    XLINE_RE = re.compile(r"^(?P<x>(?:×\s*){3,})(?P<rest>.*)$")

    chunks: List[Dict[str, Any]] = []
    in_char = False
    in_story = False

    scene_id: Optional[str] = None
    pending_speaker: Optional[str] = None

    narration_buffer = ""
    quote_buffer: Optional[Dict[str, Any]] = None  # {"speaker":..., "speaker_note":..., "text":..., "paragraph_index":...}

    chunk_seq = 0

    def add_chunk(paragraph_index: int, text: str, kind: str,
                  scene_id_: Optional[str], speaker: Optional[str],
                  speaker_note: Optional[str] = None,
                  is_heading: bool = False):
        nonlocal chunk_seq
        chunk_seq += 1
        chunks.append({
            "chunk_id": f"c{chunk_seq:06d}",
            "paragraph_index": paragraph_index,
            "text": text,
            "kind": kind,
            "scene_id": scene_id_,
            "speaker": speaker,
            "speaker_note": speaker_note,
            "is_heading": is_heading,
        })

    def flush_narration(paragraph_index_for_meta: int):
        """narration_buffer を句点で確定させる。残りは保持。"""
        nonlocal narration_buffer
        buf = norm(narration_buffer)
        if not buf:
            narration_buffer = ""
            return

        out = []
        tmp = ""
        for ch in buf:
            tmp += ch
            if ch == "。":
                out.append(tmp.strip())
                tmp = ""
        for sent in out:
            add_chunk(paragraph_index_for_meta, sent, kind="narration", scene_id_=scene_id, speaker=None)
        narration_buffer = tmp.strip()

    def flush_narration_force(paragraph_index_for_meta: int):
        """句点が無くても narration_buffer をその場で確定させる。"""
        nonlocal narration_buffer
        buf = norm(narration_buffer)
        if not buf:
            narration_buffer = ""
            return
        add_chunk(paragraph_index_for_meta, buf, kind="narration", scene_id_=scene_id, speaker=None)
        narration_buffer = ""

    def emit_dialogue(paragraph_index: int, speaker: Optional[str], quote_text: str, speaker_note: Optional[str] = None):
        """「…」を dialogue として吐く（分割しない運用なら split_dialogue_sentences は1要素だけ返す）"""
        for q in split_dialogue_sentences(quote_text):
            add_chunk(paragraph_index, q, kind="dialogue", scene_id_=scene_id, speaker=speaker, speaker_note=speaker_note)

    def extract_speaker_from_prefix(prefix: str) -> Tuple[Optional[str], Optional[str]]:
        """
        「の直前prefixが話者ラベルなら、(speaker, note) を返す。
        例: "モルデア（オフ）" / "ベリル（背）" / "子供たち"
        """
        sp, nt = try_extract_inline_speaker_prefix(prefix + "「", speaker_set)
        return sp, nt

    def process_text_stream(paragraph_index: int, text: str):
        """
        ストリーム処理：
          - quote_buffer があれば閉じるまで継続
          - quote外は narration_buffer に入れて句点で吐く
          - 「が来たら、それまでの narration を強制確定
          - prefix が話者なら narration に入れず speaker として消費
        """
        nonlocal narration_buffer, quote_buffer, pending_speaker

        t = text
        pos = 0

        while pos < len(t):
            # 1) すでに quote_buffer がある（前段落からの続き）
            if quote_buffer is not None:
                close_idx = t.find("」", pos)
                if close_idx == -1:
                    quote_buffer["text"] += norm(t[pos:])
                    return
                else:
                    quote_buffer["text"] += norm(t[pos:close_idx + 1])
                    emit_dialogue(
                        quote_buffer["paragraph_index"],
                        quote_buffer["speaker"],
                        quote_buffer["text"],
                        speaker_note=quote_buffer.get("speaker_note"),
                    )
                    quote_buffer = None
                    pos = close_idx + 1
                    continue

            # 2) 次の「 を探す
            open_idx = t.find("「", pos)
            if open_idx == -1:
                narration_buffer += norm(t[pos:])
                flush_narration(paragraph_index)
                return

            # 3) セリフ開始前に narration を強制確定
            if norm(narration_buffer):
                flush_narration_force(paragraph_index)

            # 4) 「の直前 prefix を処理
            temp_pending = None
            temp_note = None

            if open_idx > pos:
                prefix = norm(t[pos:open_idx])

                sp, nt = extract_speaker_from_prefix(prefix)
                if sp:
                    # 話者として消費（narrationには入れない）
                    temp_pending = sp
                    temp_note = nt
                else:
                    narration_buffer += prefix
                    flush_narration(paragraph_index)

            # 5) speaker を決定（優先：段落先頭〜「直前）
            prefix_for_inline = norm(t[:open_idx])
            inline_sp, inline_note = try_extract_inline_speaker_prefix(prefix_for_inline + "「", speaker_set)

            speaker_for_quote = inline_sp or temp_pending or pending_speaker
            speaker_note_for_quote = inline_note or temp_note

            # 6) 同段落内で閉じるかチェック
            close_idx = t.find("」", open_idx)
            if close_idx == -1:
                quote_buffer = {
                    "speaker": speaker_for_quote,
                    "speaker_note": speaker_note_for_quote,
                    "text": norm(t[open_idx:]),
                    "paragraph_index": paragraph_index,
                }
                return
            else:
                quote_text = norm(t[open_idx:close_idx + 1])
                emit_dialogue(paragraph_index, speaker_for_quote, quote_text, speaker_note=speaker_note_for_quote)
                pos = close_idx + 1
                continue

    for p in paras:
        idx = p["paragraph_index"]
        t = p["text"]

        # --- character section ---
        if t == HEADING_CHAR_SECTION:
            flush_narration(idx)
            in_char = True
            in_story = False
            pending_speaker = None
            add_chunk(idx, t, kind="character_heading", scene_id_=None, speaker=None, is_heading=True)
            continue

        if in_char:
            if t == CHAR_SECTION_END_TOKEN:
                in_char = False
                in_story = True
                pending_speaker = None
                add_chunk(idx, t, kind="character_end", scene_id_=None, speaker=None, is_heading=True)
                continue
            add_chunk(idx, t, kind="character", scene_id_=None, speaker=None)
            continue

        # --- meta before story ---
        if not in_story:
            add_chunk(idx, t, kind="meta", scene_id_=None, speaker=None)
            continue

        # --- story header / note ---
        if CIRCLED_NUM_RE.match(t):
            flush_narration(idx)
            if quote_buffer is not None:
                # 保険：閉じ忘れがあれば閉じて吐く
                emit_dialogue(
                    quote_buffer["paragraph_index"],
                    quote_buffer["speaker"],
                    quote_buffer["text"] + "」",
                    speaker_note=quote_buffer.get("speaker_note"),
                )
                quote_buffer = None

            scene_id = t[0]
            pending_speaker = None
            add_chunk(idx, t, kind="scene_header", scene_id_=scene_id, speaker=None, is_heading=True)
            continue

        if NOTE_RE.match(t):
            flush_narration(idx)
            pending_speaker = None
            add_chunk(idx, t, kind="note", scene_id_=scene_id, speaker=None)
            continue

        # --- NEW: marker 《...》 ---
        m = MARKER_RE.match(t)
        if m:
            flush_narration(idx)
            if quote_buffer is not None:
                emit_dialogue(
                    quote_buffer["paragraph_index"],
                    quote_buffer["speaker"],
                    quote_buffer["text"] + "」",
                    speaker_note=quote_buffer.get("speaker_note"),
                )
                quote_buffer = None

            tag = norm(m.group("tag"))
            rest = norm(m.group("rest"))

            add_chunk(idx, tag, kind="marker", scene_id_=scene_id, speaker=None)

            if rest:
                process_text_stream(idx, rest)
            continue

        # --- NEW: separator × × × ---
        mx = XLINE_RE.match(t)
        if mx:
            flush_narration(idx)
            if quote_buffer is not None:
                emit_dialogue(
                    quote_buffer["paragraph_index"],
                    quote_buffer["speaker"],
                    quote_buffer["text"] + "」",
                    speaker_note=quote_buffer.get("speaker_note"),
                )
                quote_buffer = None

            xpart = norm(mx.group("x"))
            rest = norm(mx.group("rest"))

            add_chunk(idx, xpart, kind="separator", scene_id_=scene_id, speaker=None)

            if rest:
                process_text_stream(idx, rest)
            continue

        # --- speaker-only line（表示しないで保持だけ） ---
        if ("「" not in t) and ("」" not in t) and is_speaker_line(t, speaker_set):
            pending_speaker = normalize_speaker_label(t)
            continue

        # --- main stream processing ---
        process_text_stream(idx, t)

    # end flush
    if paras:
        last_idx = paras[-1]["paragraph_index"]
        flush_narration(last_idx)
        if norm(narration_buffer):
            add_chunk(last_idx, norm(narration_buffer), kind="narration", scene_id_=scene_id, speaker=None)
        if quote_buffer is not None:
            emit_dialogue(
                quote_buffer["paragraph_index"],
                quote_buffer["speaker"],
                quote_buffer["text"] + "」",
                speaker_note=quote_buffer.get("speaker_note"),
            )
            quote_buffer = None

    return chunks


# -------------------------
# Write helpers
# -------------------------
def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Main
# -------------------------
def main():
    docx_path = CFG["script_docx"]
    if not os.path.isfile(docx_path):
        raise FileNotFoundError(f"Script DOCX not found: {docx_path}")

    paras = read_docx_paragraphs(docx_path)

    # 1) lexicon
    lexicon = build_lexicon_entities(paras)
    speaker_set = build_speaker_set_from_lexicon(lexicon)

    # 2) chunks (use speaker_set)
    chunks = build_script_chunks(paras, speaker_set)

    lex_path = os.path.join(CFG["out_dir"], CFG["lexicon_filename"])
    chunks_path = os.path.join(CFG["out_dir"], CFG["chunks_filename"])

    write_json(lex_path, lexicon)
    write_jsonl(chunks_path, chunks)

    print(f"[OK] wrote {lex_path} (entities={lexicon['stats']['entity_count']})")
    print(f"[OK] wrote {chunks_path} (chunks={len(chunks)})")
    print(f"[OK] speaker_set_size={len(speaker_set)}")


if __name__ == "__main__":
    main()
    
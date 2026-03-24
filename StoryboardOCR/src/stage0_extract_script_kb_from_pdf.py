#!/usr/bin/env python3
# extract_script_kb_from_pdf.py
# ------------------------------------------------------------
# Extract:
# 1) lexicon_entities.json  (character list)
# 2) script_chunks.jsonl    (scene/dialogue/narration chunks)
#
# Output format is kept compatible with your existing pipeline:
# - lexicon_entities.json:
#     { "episode_id": ..., "entities": [ { "type":"character", "canonical":... }, ... ], "meta": {...} }
# - script_chunks.jsonl:
#     each line has at least:
#       { "chunk_id", "paragraph_index", "kind", "scene_id", "speaker", "speaker_note", "text", ... }
#
# Heuristics:
# - Character list is extracted from the "登場人物" section (until the first scene marker ○/〇).
# - Script chunking starts after the first scene marker ○/〇.
# - Dialogues are detected by Japanese quotes 「...」 and may span multiple lines.
# - Scene headers are lines starting with ○ or 〇.
#
# Usage:
#   python extract_script_kb_from_pdf.py \
#     --pdf_path data/LWA_scenario.pdf \
#     --episode_id episode01 \
#     --script_phase script_phase1 \
#     --app_root /app
# ------------------------------------------------------------

import os
import re
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

# -------------------------
# Defaults (edit if you want)
# -------------------------
DEFAULTS = {
    "app_root": "/app",
    "episode_id": "trigger",
    "script_phase": "script",
    "pdf_path": "data/trigger/script.pdf",
    # scene marker
    "scene_markers": ["○", "〇"],
}

# -------------------------
# Regex
# -------------------------
RE_SCENE = re.compile(r"^\s*[○〇]\s*(.+?)\s*$")
RE_DIALOGUE_START = re.compile(r"^(?P<speaker>.*?)「(?P<rest>.*)$")
RE_DIALOGUE_END = re.compile(r"^(?P<body>.*)」(?P<after>.*)$")
RE_PAREN = re.compile(r"（([^）]+)）")
RE_CHAR_LINE = re.compile(r"^\s*(?P<canon>[^（(]+?)\s*(?:[（(](?P<alias>[^）)]+)[）)])?\s*$")


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------
# PDF text extraction
# -------------------------
def extract_pages_text(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Return [(page_no(1-based), text), ...]
    Tries pdfplumber first, falls back to PyMuPDF (fitz).
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    # 1) pdfplumber
    try:
        import pdfplumber  # type: ignore
        out: List[Tuple[int, str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                out.append((i, t))
        return out
    except Exception as e:
        print(f"[WARN] pdfplumber failed, fallback to fitz: {e}")

    # 2) fitz fallback
    try:
        import fitz  # type: ignore
        doc = fitz.open(pdf_path)
        out2: List[Tuple[int, str]] = []
        for i in range(len(doc)):
            page = doc[i]
            t = page.get_text("text") or ""
            out2.append((i + 1, t))
        return out2
    except Exception as e:
        raise RuntimeError(f"Both pdfplumber and fitz failed: {e}")


def build_line_list(pages_text: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """
    Flatten PDF pages into line list:
      [{"page": 1, "line": "...", "line_index": 0}, ...]
    """
    lines: List[Dict[str, Any]] = []
    idx = 0
    for page_no, text in pages_text:
        for ln in (text or "").splitlines():
            ln2 = ln.rstrip("\n")
            lines.append({"page": page_no, "line": ln2, "line_index": idx})
            idx += 1
    return lines


# -------------------------
# Character lexicon extraction
# -------------------------
def find_character_block(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract lines between "登場人物" and the first scene marker (○/〇).
    """
    start = None
    for i, obj in enumerate(lines):
        if (obj["line"] or "").strip() == "登場人物":
            start = i + 1
            break
    if start is None:
        return []

    block = []
    for j in range(start, len(lines)):
        ln = (lines[j]["line"] or "").strip()
        if not ln:
            continue
        if RE_SCENE.match(ln):
            break
        # stop if another major section-like appears (optional)
        block.append(lines[j])
    return block


def parse_character_lines(char_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse canonical + alias from lines like:
      風見あつこ（アッコ）
    """
    entities: List[Dict[str, Any]] = []
    seen = set()

    for obj in char_lines:
        raw = (obj["line"] or "").strip()
        if not raw:
            continue

        m = RE_CHAR_LINE.match(raw)
        if not m:
            continue

        canon = (m.group("canon") or "").strip()
        alias = (m.group("alias") or "").strip() if m.group("alias") else ""

        if not canon:
            continue
        if canon in seen:
            continue

        seen.add(canon)
        aliases: List[str] = []
        if alias:
            # allow multiple aliases split by separators if present
            # (keep conservative)
            for a in re.split(r"[・/／,，\s]+", alias):
                a2 = a.strip()
                if a2:
                    aliases.append(a2)

        entities.append({
            "id": f"ch{len(entities)+1:04d}",
            "type": "character",
            "canonical": canon,
            "aliases": aliases,
            "source": {
                "kind": "pdf_character_list",
                "page": obj["page"],
                "raw_line": raw,
            }
        })

    return entities


# -------------------------
# Script chunking
# -------------------------
@dataclass
class ChunkCtx:
    scene_idx: int
    scene_id: Optional[str]
    scene_title: Optional[str]
    next_chunk_num: int
    paragraph_index: int


def is_noise_line(ln: str) -> bool:
    t = (ln or "").strip()
    if not t:
        return True
    # common page-number-only lines
    if t.isdigit():
        return True
    return False


def clean_speaker(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract speaker and speaker_note from something like:
      アッコ（M）  -> speaker="アッコ", speaker_note="M"
      老教師（off でセリフ先行） -> speaker="老教師", speaker_note="off でセリフ先行"
    """
    s = (s or "").strip()
    if not s:
        return None, None
    note = None
    m = RE_PAREN.search(s)
    if m:
        note = (m.group(1) or "").strip() or None
        s2 = RE_PAREN.sub("", s).strip()
        s = s2
    s = re.sub(r"\s+", "", s)  # speaker usually no spaces
    return (s or None), note


def write_chunk_jsonl(fp, obj: Dict[str, Any]):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def chunk_script(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert line list into chunk dicts (kept in-memory, then written to jsonl).
    """
    chunks: List[Dict[str, Any]] = []
    ctx = ChunkCtx(scene_idx=0, scene_id=None, scene_title=None, next_chunk_num=1, paragraph_index=0)

    # Start chunking only after the first scene marker appears
    script_started = False

    narr_buf: List[Dict[str, Any]] = []

    def flush_narration():
        nonlocal narr_buf
        if not narr_buf:
            return
        # join narration lines
        text = "\n".join([(o["line"] or "").rstrip() for o in narr_buf]).strip()
        if text:
            ch = {
                "chunk_id": f"c{ctx.next_chunk_num:06d}",
                "paragraph_index": ctx.paragraph_index,
                "kind": "narration",
                "scene_id": ctx.scene_id,
                "scene_title": ctx.scene_title,
                "speaker": None,
                "speaker_note": None,
                "text": text,
                "source": {
                    "page_start": narr_buf[0]["page"],
                    "page_end": narr_buf[-1]["page"],
                    "line_index_start": narr_buf[0]["line_index"],
                    "line_index_end": narr_buf[-1]["line_index"],
                },
            }
            chunks.append(ch)
            ctx.next_chunk_num += 1
            ctx.paragraph_index += 1
        narr_buf = []

    i = 0
    while i < len(lines):
        obj = lines[i]
        page = obj["page"]
        ln = obj["line"] or ""

        if is_noise_line(ln):
            i += 1
            continue

        # detect script start
        if RE_SCENE.match(ln):
            script_started = True

        if not script_started:
            i += 1
            continue

        # scene header
        m_scene = RE_SCENE.match(ln)
        if m_scene:
            flush_narration()
            title = (m_scene.group(1) or "").strip()
            ctx.scene_idx += 1
            ctx.scene_id = f"S{ctx.scene_idx:03d}"
            ctx.scene_title = title

            ch = {
                "chunk_id": f"c{ctx.next_chunk_num:06d}",
                "paragraph_index": ctx.paragraph_index,
                "kind": "scene",
                "scene_id": ctx.scene_id,
                "scene_title": ctx.scene_title,
                "speaker": None,
                "speaker_note": None,
                "text": title,
                "source": {"page": page, "line_index": obj["line_index"], "raw_line": ln.strip()},
            }
            chunks.append(ch)
            ctx.next_chunk_num += 1
            ctx.paragraph_index += 1
            i += 1
            continue

        # dialogue start?
        # (must contain 「 somewhere; allow leading speaker + quote)
        m_dlg = RE_DIALOGUE_START.match(ln)
        if m_dlg and "「" in ln:
            flush_narration()

            speaker_raw = (m_dlg.group("speaker") or "").strip()
            speaker, speaker_note = clean_speaker(speaker_raw)

            # start collecting dialogue body from after first 「
            rest = (m_dlg.group("rest") or "")
            dlg_parts = [rest]

            # if closing 」 is in the same line, finish immediately
            if "」" in rest:
                joined = rest
                body = joined.split("」", 1)[0]
                dlg_text_with_quotes = f"「{body}」"
                ch = {
                    "chunk_id": f"c{ctx.next_chunk_num:06d}",
                    "paragraph_index": ctx.paragraph_index,
                    "kind": "dialogue",
                    "scene_id": ctx.scene_id,
                    "scene_title": ctx.scene_title,
                    "speaker": speaker,
                    "speaker_note": speaker_note,
                    "text": dlg_text_with_quotes,      # compatibility: includes quotes
                    "text_no_quotes": body,            # useful for retrieval if you want
                    "source": {"page": page, "line_index": obj["line_index"], "raw_line": ln.strip()},
                }
                chunks.append(ch)
                ctx.next_chunk_num += 1
                ctx.paragraph_index += 1
                i += 1
                continue

            # multi-line dialogue: scan until the first closing 」
            j = i + 1
            end_found = False
            end_obj = obj
            while j < len(lines):
                obj2 = lines[j]
                ln2 = obj2["line"] or ""
                if is_noise_line(ln2):
                    j += 1
                    continue
                dlg_parts.append(ln2)
                if "」" in ln2:
                    end_found = True
                    end_obj = obj2
                    break
                j += 1

            joined = "\n".join(dlg_parts)
            if "」" in joined:
                body = joined.split("」", 1)[0]
            else:
                # if not found, keep everything (rare)
                body = joined

            dlg_text_with_quotes = f"「{body}」"
            ch = {
                "chunk_id": f"c{ctx.next_chunk_num:06d}",
                "paragraph_index": ctx.paragraph_index,
                "kind": "dialogue",
                "scene_id": ctx.scene_id,
                "scene_title": ctx.scene_title,
                "speaker": speaker,
                "speaker_note": speaker_note,
                "text": dlg_text_with_quotes,
                "text_no_quotes": body,
                "source": {
                    "page_start": obj["page"],
                    "page_end": end_obj["page"],
                    "line_index_start": obj["line_index"],
                    "line_index_end": end_obj["line_index"],
                },
            }
            chunks.append(ch)
            ctx.next_chunk_num += 1
            ctx.paragraph_index += 1

            i = j + 1
            continue

        # otherwise narration line
        narr_buf.append(obj)
        i += 1

    flush_narration()
    return chunks


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app_root", default=DEFAULTS["app_root"])
    ap.add_argument("--episode_id", default=DEFAULTS["episode_id"])
    ap.add_argument("--script_phase", default=DEFAULTS["script_phase"])
    ap.add_argument("--pdf_path", default=DEFAULTS["pdf_path"])
    args = ap.parse_args()

    app_root = args.app_root
    episode_id = args.episode_id
    script_phase = args.script_phase

    pdf_path = args.pdf_path
    # allow relative path from app_root
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(app_root, pdf_path)

    out_root = os.path.join(app_root, "outputs", script_phase, episode_id)
    ensure_dir(out_root)

    lexicon_path = os.path.join(out_root, "lexicon_entities.json")
    chunks_path = os.path.join(out_root, "script_chunks.jsonl")

    pages_text = extract_pages_text(pdf_path)
    lines = build_line_list(pages_text)

    # ---- lexicon ----
    char_block = find_character_block(lines)
    entities = parse_character_lines(char_block)

    lexicon = {
        "episode_id": episode_id,
        "entities": entities,
        "meta": {
            "created_at": now_iso_utc(),
            "source_pdf": pdf_path,
            "script_phase": script_phase,
            "extractor": "extract_script_kb_from_pdf.py",
            "character_count": len(entities),
        }
    }

    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {lexicon_path} (characters={len(entities)})")

    # ---- chunks ----
    chunks = chunk_script(lines)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            write_chunk_jsonl(f, ch)
    print(f"[OK] wrote {chunks_path} (chunks={len(chunks)})")

    # Optional small stats (console)
    kinds = {}
    for ch in chunks:
        kinds[ch["kind"]] = kinds.get(ch["kind"], 0) + 1
    print("[INFO] chunk kinds:", kinds)


if __name__ == "__main__":
    main()

"""Unstructured side: download regulatory PDFs, extract text page by page, clean it and chunk it.

Each chunk keeps its document id and page number, so every answer can cite exactly where a fact came from.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from pypdf import PdfReader

from ..config import Settings

# SEBI letterheads carry a Hindi line that extracts as mojibake, plus page furniture we don't want to embed.
_NOISE = [
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.I),
    re.compile(r"^\s*Securities and Exchange Board of India\s*$", re.I),
    re.compile(r"^\s*\d+\s*$"),
]


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    page: int
    text: str


def _is_noise(line: str) -> bool:
    if any(p.match(line) for p in _NOISE):
        return True
    letters = [c for c in line if c.isalpha()]
    # mostly non-ASCII letters = mis-encoded Devanagari header
    return bool(letters) and sum(1 for c in letters if ord(c) > 127) / len(letters) > 0.5


def clean_page(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    kept = [ln for ln in lines if ln and not _is_noise(ln)]
    out = " ".join(kept)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"(\w)- (\w)", r"\1\2", out)  # undo hyphenation across line breaks
    out = re.sub(r"([a-z]{3,})(\d)", r"\1 \2", out)  # table cells glued together: "companies65%"
    return out.strip()


def chunk_words(words: list[str], size: int = 180, overlap: int = 40) -> list[list[str]]:
    if not words:
        return []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def chunk_document(doc_id: str, title: str, pages: list[str], size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Chunk within a page (never across pages) so a citation's page number is always exact."""
    chunks: list[Chunk] = []
    for page_no, raw in enumerate(pages, start=1):
        words = clean_page(raw).split()
        for i, part in enumerate(chunk_words(words, size, overlap)):
            if len(part) < 12:  # skip slivers (signature blocks, stray headers)
                continue
            chunks.append(Chunk(f"{doc_id}:p{page_no}:{i}", doc_id, title, page_no, " ".join(part)))
    return chunks


def download(settings: Settings) -> list[Path]:
    sources = json.loads(settings.sources_file.read_text())["documents"]
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "FundLens/0.1 (research)"}) as client:
        for src in sources:
            path = settings.docs_dir / f"{src['id']}.pdf"
            if not path.exists():
                resp = client.get(src["url"])
                resp.raise_for_status()
                path.write_bytes(resp.content)
            paths.append(path)
    return paths


def build_chunks(settings: Settings) -> list[Chunk]:
    sources = json.loads(settings.sources_file.read_text())["documents"]
    chunks: list[Chunk] = []
    for src in sources:
        reader = PdfReader(settings.docs_dir / f"{src['id']}.pdf")
        pages = [p.extract_text() or "" for p in reader.pages]
        chunks.extend(chunk_document(src["id"], src["title"], pages))
    with settings.chunks_file.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    return chunks


def load_chunks(settings: Settings) -> list[Chunk]:
    with settings.chunks_file.open() as f:
        return [Chunk(**json.loads(line)) for line in f]

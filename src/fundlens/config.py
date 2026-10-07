"""Runtime configuration, read from environment variables (see .env.example).

Nothing secret is needed for the default local setup. If a hosted LLM is plugged in later, its key is read
from the environment only and is never logged or written to traces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("FUNDLENS_DATA_DIR", ROOT / "data")))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("FUNDLENS_MODEL", "llama3.2:3b"))
    embed_model: str = field(default_factory=lambda: os.getenv("FUNDLENS_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("FUNDLENS_MAX_STEPS", "4")))
    top_k: int = field(default_factory=lambda: int(os.getenv("FUNDLENS_TOP_K", "5")))
    trace_file: Path = field(
        default_factory=lambda: Path(os.getenv("FUNDLENS_TRACE_FILE", ROOT / "data" / "traces.jsonl"))
    )

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def chunks_file(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_file(self) -> Path:
        return self.data_dir / "funds.sqlite"

    @property
    def nav_snapshot(self) -> Path:
        return self.data_dir / "nav_snapshot.csv.gz"

    @property
    def sources_file(self) -> Path:
        return self.data_dir / "sources.json"


def get_settings() -> Settings:
    return Settings()

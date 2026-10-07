"""HTTP API + web UI.

POST /ask      {"question": "..."} -> answer, citations, evidence, trace
GET  /metrics  latency p50/p95, tokens, refusal rate, citation rate, tool usage (from the trace log)
GET  /health   model and index status
GET  /         chat UI
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import tracing
from .agent import Agent
from .config import get_settings
from .llm import OllamaClient
from .retrieval.index import HybridIndex
from .tools.registry import Toolbox

WEB = Path(__file__).parent / "web"

app = FastAPI(title="FundLens", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    s = get_settings()
    return Agent(s, OllamaClient(s.ollama_url, s.model), Toolbox(s, HybridIndex.load(s)))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    agent = get_agent()
    try:
        result = agent.ask(req.question)
    except Exception as e:  # model server down, timeouts
        raise HTTPException(status_code=503, detail=f"model unavailable: {e}") from e
    tracing.record(get_settings().trace_file, result)
    return result.to_dict()


@app.get("/metrics")
def metrics() -> dict:
    return tracing.summarise(get_settings().trace_file)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"model": s.model, "index": (s.index_dir / "vectors.npy").exists(), "db": s.db_file.exists()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")

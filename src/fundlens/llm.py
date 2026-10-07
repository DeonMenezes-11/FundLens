"""LLM clients behind one small interface, so the agent doesn't care which provider runs underneath.

OllamaClient talks to a local model (no data leaves the machine, no API key). ScriptedLLM replays fixed
responses for deterministic tests. A hosted provider (e.g. Azure OpenAI) would be one more class here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply: ...


class OllamaClient:
    def __init__(self, url: str, model: str, temperature: float = 0.0, timeout: float = 180.0):
        self.url = url.rstrip("/")
        self.name = model
        self.temperature = temperature
        self._http = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": 7, "num_ctx": 8192},
        }
        if tools:
            payload["tools"] = tools
        resp = self._http.post(f"{self.url}/api/chat", json=payload)
        resp.raise_for_status()
        body = resp.json()
        msg = body.get("message", {})
        calls = [
            ToolCall(c["function"]["name"], c["function"].get("arguments") or {}) for c in msg.get("tool_calls") or []
        ]
        return LLMReply(
            content=msg.get("content") or "",
            tool_calls=calls,
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
        )


class ScriptedLLM:
    """Returns pre-baked replies in order and records what it was sent. For tests only."""

    def __init__(self, replies: list[LLMReply]):
        self.name = "scripted"
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None) -> LLMReply:
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            return LLMReply("I could not find this in the available sources.")
        return self.replies.pop(0)

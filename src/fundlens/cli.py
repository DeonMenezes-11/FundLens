"""Command line: build data, ask questions, serve the app, run the evaluation.

    fundlens setup            download documents, build chunks, embeddings and the NAV database
    fundlens refresh-nav      pull the latest NAVs into a new snapshot
    fundlens ask "question"   one-off question from the terminal
    fundlens serve            start the API + UI on http://localhost:8000
"""
from __future__ import annotations

import argparse
import json

from .config import get_settings


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fundlens")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("refresh-nav")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    s = get_settings()

    if args.cmd == "setup":
        from .ingest import documents, nav
        from .retrieval.index import HybridIndex

        print("downloading documents...")
        documents.download(s)
        chunks = documents.build_chunks(s)
        print(f"{len(chunks)} chunks")
        if not s.nav_snapshot.exists():
            print(f"{nav.fetch_snapshot(s)} NAV rows fetched")
        nav.build_db(s)
        print("embedding chunks (first run downloads the embedding model)...")
        HybridIndex.build(s)
        print("done")
    elif args.cmd == "refresh-nav":
        from .ingest import nav

        print(f"{nav.fetch_snapshot(s)} NAV rows")
        nav.build_db(s)
    elif args.cmd == "ask":
        from .agent import Agent
        from .llm import OllamaClient
        from .retrieval.index import HybridIndex
        from .tools.registry import Toolbox

        agent = Agent(s, OllamaClient(s.ollama_url, s.model), Toolbox(s, HybridIndex.load(s)))
        r = agent.ask(args.question)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2, default=str))
        else:
            print(r.answer)
            print(f"\n[{r.latency_ms / 1000:.1f}s · tools: {', '.join(r.tools_used) or 'none'} · flags: {r.flags}]")
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("fundlens.api:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

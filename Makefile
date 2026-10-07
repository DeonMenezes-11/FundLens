PY ?= .venv/bin/python

setup:            ## download SEBI docs, build chunks, embeddings and NAV database
	$(PY) -m fundlens.cli setup
serve:            ## API + UI on http://localhost:8000
	$(PY) -m fundlens.cli serve
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests eval
eval:             ## full agent evaluation (needs Ollama running)
	$(PY) -m eval.run_eval
eval-retrieval:   ## retrieval-only evaluation (no LLM)
	$(PY) -m eval.run_eval --retrieval-only

-include .env.local
export

.PHONY: run build-db test lint

run:
	uv run python -m ki_kompass_rag.main

build-db:
	uv run python -m ki_kompass_rag.build_app_db

test:
	uv run pytest -q

lint:
	uv run ruff check
	uv run ruff format --check
	uv run mypy conversational-toolkit/src backend/src

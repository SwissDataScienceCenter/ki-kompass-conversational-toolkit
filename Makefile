include .env.local
export

run:
	BACKEND=openai uv run python -m ki_kompass_rag.main

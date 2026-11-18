include .env.local
export

run:
	BACKEND=openai python -m ki_kompass_rag.main

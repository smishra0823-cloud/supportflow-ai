.PHONY: demo app api test docker-up docker-down

demo:
	.venv/bin/python -m scripts.start_demo

app:
	.venv/bin/streamlit run app.py --server.port 8503

api:
	.venv/bin/python -m scripts.run_api

test:
	.venv/bin/pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down

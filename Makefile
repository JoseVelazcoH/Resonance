.PHONY: help install env backend frontend dev test test-laya

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[32m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

env: ## Create backend/.env and frontend/.env from the examples (never overwrites)
	@test -f backend/.env || cp backend/.env.example backend/.env
	@test -f frontend/.env || cp frontend/.env.example frontend/.env

install: env ## Install backend (uv sync) and frontend (npm install) dependencies
	cd backend && uv sync
	cd frontend && npm install

backend: ## Run the API on http://127.0.0.1:8000
	cd backend && uv run uvicorn mood_dj.api.main:app --host 127.0.0.1 --port 8000

frontend: ## Run the web app on http://127.0.0.1:5173
	cd frontend && npm run dev

dev: ## Run backend and frontend together
	$(MAKE) -j2 backend frontend

test: ## Unit tests, no network, no model
	cd backend && uv run pytest

test-laya: ## Integration tests against the real Laya model
	cd backend && uv run pytest -m laya

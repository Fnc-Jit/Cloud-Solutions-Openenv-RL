.PHONY: test docker docker-standalone validate push clean help

IMAGE_NAME = cloudfinops-env
IMAGE_TAG = latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Run unit tests
	PYTHONPATH=. pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	PYTHONPATH=. pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

docker: ## Build Docker image (Meta base image)
	docker build -f server/Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-standalone: ## Build Docker image (python:3.11-slim, no external deps)
	docker build -f server/Dockerfile.standalone -t $(IMAGE_NAME):$(IMAGE_TAG) .

run: ## Start the environment server locally
	docker run --env-file .env -p 8000:8000 $(IMAGE_NAME):$(IMAGE_TAG)

run-local: ## Start the server locally with uvicorn
	uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

inference: ## Run the baseline evaluator against local server
	python inference.py

inference-docker: ## Run inference inside Docker
	docker run --env-file .env -e ENV_BASE_URL=http://host.docker.internal:8000 $(IMAGE_NAME):$(IMAGE_TAG) python inference.py

validate: ## Run pre-submission validation
	python pre_validation.py

smoke: ## Smoke test — build, start, hit endpoints
	docker build -f server/Dockerfile.standalone -t $(IMAGE_NAME):smoke .
	docker run -d --name smoke-test -p 8000:8000 $(IMAGE_NAME):smoke
	@sleep 5
	@echo "GET /"
	@curl -sf http://localhost:8000/ | python -m json.tool
	@echo "GET /health"
	@curl -sf http://localhost:8000/health | python -m json.tool
	@echo "POST /reset"
	@curl -sf -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}' | python -m json.tool
	@docker stop smoke-test && docker rm smoke-test
	@echo "Smoke test passed."

push: ## Push to Hugging Face Spaces
	openenv push

lint: ## Run syntax check on all Python files
	python -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('**/*.py', recursive=True) if 'venv' not in f]"

clean: ## Remove Docker images and caches
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	docker rmi $(IMAGE_NAME):smoke 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

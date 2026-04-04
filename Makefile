.PHONY: help test validate docker docker-standalone run push clean

IMAGE_NAME = cloudfinops-env
ENV_URL = http://localhost:8000

help:
	@echo "CloudFinOps-Env — Makefile targets:"
	@echo ""
	@echo "  make test              Run pytest suite"
	@echo "  make validate          Run pre-submission validation"
	@echo "  make docker            Build Docker image (openenv-base)"
	@echo "  make docker-standalone Build Docker image (python:3.11-slim)"
	@echo "  make run               Start server via uvicorn"
	@echo "  make run-docker        Run container on port 8000"
	@echo "  make push              Deploy to Hugging Face Spaces"
	@echo "  make clean             Remove build artifacts"

test:
	python3 -m pytest tests/ -v --tb=short

validate:
	python3 pre_validation.py

validate-docker:
	python3 pre_validation.py --docker

docker:
	docker build -f server/Dockerfile -t $(IMAGE_NAME) .

docker-standalone:
	docker build -f server/Dockerfile.standalone -t $(IMAGE_NAME) .

run:
	uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

run-docker:
	docker run --env-file .env -p 8000:8000 $(IMAGE_NAME)

push:
	openenv push

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ build/ dist/ *.egg-info/

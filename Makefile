.PHONY: all lint test validate-data prepare train validate registry api docker clean help

all: lint test validate-data prepare train validate
	@echo "✓ Pipeline local aprobado. Ejecuta 'make docker' para construir la API."

lint:
	flake8 src/ api/ tests/ --config=setup.cfg

test:
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

validate-data:
	python -m src.validate_data

prepare:
	python -m src.prepare_data

train:
	python -m src.train_pipeline --n-iter 6 --cv-splits 3

validate:
	python -m src.validate_model

registry:
	python -m src.register_model

api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -t seguro-antirobos-api:local .
	docker run --rm -d --name seguro-antirobos-smoke -p 8000:8000 seguro-antirobos-api:local
	sleep 4
	curl --fail http://localhost:8000/health
	docker rm -f seguro-antirobos-smoke

clean:
	rm -rf artifacts/* data/processed/* mlruns/ mlflow.db .coverage htmlcov/ .pytest_cache/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

help:
	@echo "make all | lint | test | validate-data | prepare | train | validate | registry | api | docker | clean"

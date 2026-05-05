PYTHON=poetry run python

.PHONY: setup pipeline test lint format clean cache all

setup:
	poetry install

pipeline:
	$(PYTHON) -m src.diabetes_prediction.pipeline.pipeline

test:
	poetry run pytest tests -v

lint:
	poetry run ruff check src tests

format:
	poetry run black src tests

fix:
	poetry run ruff check src tests --fix	


all: format fix lint test pipeline
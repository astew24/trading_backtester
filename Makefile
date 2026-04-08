PYTHON ?= python3

.PHONY: install install-dev test lint format typecheck run

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

format:
	black .

typecheck:
	mypy src

run:
	$(PYTHON) main.py --config config.yaml

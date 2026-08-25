# Everything assumes you're in WSL with the venv active.
PY := PYTHONPATH=src python3

.PHONY: help setup train eval quantize demo test lint all clean firmware-model upload

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

setup: ## create venv and install deps
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train: ## simulate, extract features, train the autoencoder
	$(PY) -m motor_anomaly.train --config config/default.yaml

eval: ## score the held-out test split
	$(PY) -m motor_anomaly.evaluate

quantize: ## int8 TFLite + recalibrated threshold + C header
	$(PY) -m motor_anomaly.convert_tflite

demo: ## run the edge detector on a healthy -> fault stream
	$(PY) -m motor_anomaly.edge.runner --demo --minutes 2

upload: ## dry-run the S3 spool upload
	$(PY) -m motor_anomaly.edge.uploader

test: ## run the unit tests
	$(PY) -m pytest tests -q

lint:
	ruff check src tests && ruff format --check src tests

firmware-model: quantize ## regenerate model_data.h for the Arduino sketch
	@echo "firmware/nano33ble/model_data.h regenerated"

all: train eval quantize demo ## the full pipeline

clean:
	rm -rf artifacts/*.keras artifacts/*.tflite artifacts/*.npz artifacts/*.json artifacts/spool.jsonl
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

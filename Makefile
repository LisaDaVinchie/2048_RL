BASE_FOLDER = $(shell pwd)
PYTHON = $(shell which python3)

SRC_FOLDER = $(BASE_FOLDER)/src
TEST_FOLDER = $(BASE_FOLDER)/test
PARAMS_FILE = $(SRC_FOLDER)/params.json

.PHONY: train test

train:
	@echo "Training model..."
	$(PYTHON) $(SRC_FOLDER)/train_agent.py --params $(PARAMS_FILE)

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_FOLDER) -p "*_test.py"
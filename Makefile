BASE_FOLDER = $(shell pwd)
PYTHON = $(shell which python3)

SRC_FOLDER = $(BASE_FOLDER)/src
TEST_FOLDER = $(BASE_FOLDER)/test
DATA_FOLDER = $(BASE_FOLDER)/data
PARAMS_FILE = $(SRC_FOLDER)/params.json
PATHS_FILE = $(SRC_FOLDER)/paths.json

# Name of the final result file
EXT=.txt
FINAL_RESULT_BASENAME = results
# Find the next available filename
FINAL_RESULT_PATH=$(shell i=0; while [ -e "$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)_$$i$(EXT)" ]; do i=$$((i+1)); done; echo "$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)_$$i$(EXT)")

.PHONY: config train test help

config:
	@echo "Configuring paths..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"final_result_path\": \"$(FINAL_RESULT_PATH)\"" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

train: config
	@echo "Training model..."
	$(PYTHON) $(SRC_FOLDER)/train_agent.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_FOLDER) -p "*_test.py"

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  config: Configure paths"
	@echo "  train: Train the model"
	@echo "  test: Run tests"
	@echo "  help: Show this help message"
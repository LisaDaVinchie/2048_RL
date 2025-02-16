BASE_FOLDER = $(shell pwd)
PYTHON = $(shell which python3)

SRC_FOLDER = $(BASE_FOLDER)/src
TEST_FOLDER = $(BASE_FOLDER)/test
DATA_FOLDER = $(BASE_FOLDER)/data
FIGS_FOLDER = $(BASE_FOLDER)/figs

PARAMS_FILE = $(SRC_FOLDER)/params.json
PATHS_FILE = $(SRC_FOLDER)/paths.json
FINAL_RESULT_BASENAME = results
RESULT_FILE_EXT=.txt
FIGS_BASENAME = figure
FIGS_EXT=.png


# Name of the final result file

# Find the nRESULT_FILE_EXT available filename
FINAL_RESULT_PATH=$(shell i=0; while [ -e "$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)")

.PHONY: config train test plot help

config:
	@echo "Configuring paths..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"final_result_basename\": \"$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)\"," >> $(PATHS_FILE)
	@echo "    \"result_file_ext\": \"$(RESULT_FILE_EXT)\"," >> $(PATHS_FILE)
	@echo "    \"final_result_path\": \"$(FINAL_RESULT_PATH)\"," >> $(PATHS_FILE)
	@echo "    \"figure_basename\": \"$(FIGS_FOLDER)/$(FIGS_BASENAME)\"," >> $(PATHS_FILE)
	@echo "    \"figure_ext\": \"$(FIGS_EXT)\"" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

train: config
	@echo "Training model..."
	$(PYTHON) $(SRC_FOLDER)/train_agent.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_FOLDER) -p "*_test.py"

plot: config
	@echo "Plotting results..."
	$(PYTHON) $(SRC_FOLDER)/plot_results.py --paths $(PATHS_FILE)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  config: Configure paths"
	@echo "  train: Train the model"
	@echo "  test: Run tests"
	@echo "  help: Show this help message"
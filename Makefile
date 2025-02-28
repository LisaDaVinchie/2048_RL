BASE_FOLDER = $(shell pwd)
PYTHON = $(shell which python3)

SRC_FOLDER = $(BASE_FOLDER)/src
TEST_FOLDER = $(BASE_FOLDER)/test
DATA_FOLDER = $(BASE_FOLDER)/data
FIGS_FOLDER = $(BASE_FOLDER)/figs
WEIGHTS_FOLDER = $(DATA_FOLDER)/weights
RANDOM_RESULTS_FOLDER = $(DATA_FOLDER)/random_results

PARAMS_FILE = $(SRC_FOLDER)/params.json
PATHS_FILE = $(SRC_FOLDER)/paths.json
FINAL_RESULT_BASENAME = results
RESULT_FILE_EXT=.txt
FIGS_BASENAME = figure
FIGS_EXT=.png
GRID_FILE_BASENAME = $(DATA_FOLDER)/grid
WEIGHTS_BASENAME = $(WEIGHTS_FOLDER)/weights
WEIGHTS_EXT = .pth


# Name of the final result file

# Find the nRESULT_FILE_EXT available filename
IDX=$(shell i=0; while [ -e "$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
FINAL_RESULT_PATH = $(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)_$(IDX)$(RESULT_FILE_EXT)
GRID_FILE_PATH = $(GRID_FILE_BASENAME)_$(IDX).txt
WEIGHTS_FILE_PATH = $(WEIGHTS_BASENAME)_$(IDX)$(WEIGHTS_EXT)

RANDOM_IDX=$(shell i=0; while [ -e "$(RANDOM_RESULTS_FOLDER)/$(FINAL_RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
FINAL_RANDOM_RESULT_PATH = $(RANDOM_RESULTS_FOLDER)/$(FINAL_RESULT_BASENAME)_$(RANDOM_IDX)$(RESULT_FILE_EXT)

.PHONY: config train test plot help

config:
	@echo "Configuring paths..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"final_result_basename\": \"$(DATA_FOLDER)/$(FINAL_RESULT_BASENAME)\"," >> $(PATHS_FILE)
	@echo "    \"result_file_ext\": \"$(RESULT_FILE_EXT)\"," >> $(PATHS_FILE)
	@echo "    \"final_result_path\": \"$(FINAL_RESULT_PATH)\"," >> $(PATHS_FILE)
	@echo "    \"figure_basename\": \"$(FIGS_FOLDER)/$(FIGS_BASENAME)\"," >> $(PATHS_FILE)
	@echo "    \"figure_ext\": \"$(FIGS_EXT)\"," >> $(PATHS_FILE)
	@echo "    \"grid_file_path\": \"$(GRID_FILE_PATH)\"," >> $(PATHS_FILE)
	@echo "    \"weights_file_path\": \"$(WEIGHTS_FILE_PATH)\"," >> $(PATHS_FILE)
	@echo "    \"random_result_path\": \"$(FINAL_RANDOM_RESULT_PATH)\"" >> $(PATHS_FILE)
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
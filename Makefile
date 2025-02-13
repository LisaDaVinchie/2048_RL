BASE_FOLDER = $(shell pwd)
PYTHON = $(shell which python3)

SRC_FOLDER = $(BASE_FOLDER)/src
TEST_FOLDER = $(BASE_FOLDER)/test

.PHONY: test

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_FOLDER) -p "*_test.py"
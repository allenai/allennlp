# Makefile for Tutorials
#
SHELL := /bin/bash

# You can set these variables from the command line.
TUTORIAL_OUTPUT_DIR = build
ALLENNLP_WEBSITE_DIR = ../allennlp-website

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build             to clean and build all the tutorials"
	@echo "  clean             to wipe out the build directory"
	@echo "  getting-started   to build the getting-started tutorials"
	@echo "  notebooks         to build the jupyter notebook tutorials"

.PHONY: clean
clean:
	@echo "Cleaning out the build directory"
	@rm -rf $(TUTORIAL_OUTPUT_DIR)

.PHONY: getting-started
getting-started:
	@echo "Building the getting-started tutorials"
	@mkdir -p $(TUTORIAL_OUTPUT_DIR)
	@cp -r getting_started/*.md $(TUTORIAL_OUTPUT_DIR)/

.PHONY: notebooks
notebooks:
	@echo "Building the jupyter notebooks tutorials"
	@mkdir -p $(TUTORIAL_OUTPUT_DIR)
	@for file in notebooks/*.ipynb; do jupyter nbconvert --to markdown "$$file" --output-dir $(TUTORIAL_OUTPUT_DIR); done

.PHONY: hyphenate
hyphenate:
	@echo "Hyphenating"
	@for f in build/*.md; do if [[ $$f == *"_"* ]]; then mv "$$f" "$${f//_/-}"; fi; done

.PHONY: copy
copy:
	@echo "Copying from $(TUTORIAL_OUTPUT_DIR) to $(ALLENNLP_WEBSITE_DIR)"
	@cp -r $(TUTORIAL_OUTPUT_DIR) $(ALLENNLP_WEBSITE_DIR)/tutorials

.PHONY: build
build:
	@make clean
	@make getting-started
	@make notebooks
	@make hyphenate
	@echo "Build finished. The md pages are in $(TUTORIAL_OUTPUT_DIR)."

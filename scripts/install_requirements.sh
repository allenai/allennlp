#!/bin/bash

pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy.en.download all

# only install test requirements if explicitly specified
if [[ "$INSTALL_TEST_REQUIREMENTS" == "true" ]]; then
    pip install -r requirements_test.txt
fi

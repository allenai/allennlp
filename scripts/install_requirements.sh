#!/bin/bash

pip install -r requirements.txt
# Temporary fix to the build whilst NLTK sort stuff out. TODO(Mark): revert this.
python -m nltk.downloader -u https://pastebin.com/raw/D3TBY4Mj punkt
# python -m nltk.downloader punkt
python -m spacy.en.download all

# only install test requirements if explicitly specified
if [[ "$INSTALL_TEST_REQUIREMENTS" == "true" ]]; then
    pip install -r requirements_test.txt
fi

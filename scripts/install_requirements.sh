#!/bin/bash

pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy.en.download all

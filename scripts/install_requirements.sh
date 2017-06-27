#!/bin/bash

pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy.en.download all
pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl

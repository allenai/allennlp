#!/bin/bash

pip install -r requirements.txt
python -m nltk.downloader punkt
spacy download en_core_web_sm # Used by allennlp.data.tokenizers.word_splitter.NltkWordSplitter

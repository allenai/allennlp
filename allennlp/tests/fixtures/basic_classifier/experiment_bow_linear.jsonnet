local COMMON = import 'common.jsonnet';

{
    'dataset_reader': COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": "allennlp/tests/fixtures/data/text_classification_json/imdb_corpus.jsonl",
    "validation_data_path": "allennlp/tests/fixtures/data/text_classification_json/imdb_corpus.jsonl",
    "model": {
        "type": "basic_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "bag_of_word_counts",
                    "ignore_oov": true
                }
            }
        }
    },
    "iterator": COMMON['iterator'],
    "trainer": COMMON['trainer'],
}


{
    "dataset_reader": {
        "lazy": false,
        "type": "text_classification_json",
        "tokenizer": {
            "word_splitter": "spacy"
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "tokens",
                "lowercase_tokens": true
            }
        },
        "max_sequence_length": 400
    },
    "train_data_path": "allennlp/tests/fixtures/data/text_classification_json/imdb_corpus.jsonl",
    "validation_data_path": "allennlp/tests/fixtures/data/text_classification_json/imdb_corpus.jsonl",
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}

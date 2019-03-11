local COMMON = import 'common.jsonnet';

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": "allennlp/tests/fixtures/data/text_classification_json/imdb_corpus.jsonl",
    "validation_data_path": "allennlp/tests/fixtures/data/text_classification_json/imdb_corpus.jsonl",
    "model": {
        "type": "basic_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        },
        "seq2vec_encoder": {
           "type": "cnn",
           "num_filters": 100,
           "embedding_dim": 300,
           "output_dim": 512
        }
    },
    "iterator": COMMON['iterator'],
    "trainer": COMMON['trainer'],
}

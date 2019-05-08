local COMMON = import 'common.jsonnet';

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": COMMON['train_data_path'],
    "validation_data_path": COMMON['validation_data_path'],
    "model": {
        "type": "basic_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "trainable": true
                }
            }
        },
        "seq2seq_encoder": {
            "type": "lstm",
            "num_layers": 1,
            "bidirectional": false,
            "input_size": 10,
            "hidden_size": 16
        },
        "seq2vec_encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 16,
            "averaged": true
        },
    },
    "iterator": COMMON['iterator'],
    "trainer": COMMON['trainer']
}


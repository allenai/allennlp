local COMMON = import 'common.jsonnet';

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": COMMON['train_data_path'], 
    "validation_data_path": COMMON['train_data_path'],
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
        "seq2vec_encoder": {
           "type": "cnn",
           "num_filters": 8,
           "embedding_dim": 10,
           "output_dim": 16
        }
    },
    "iterator": COMMON['iterator'],
    "trainer": COMMON['trainer'],
}

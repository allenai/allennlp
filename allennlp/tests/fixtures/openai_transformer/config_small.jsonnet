// NER model that uses a tiny (random-weight) transformer.
{
    "dataset_reader": {
      "type": "conll2003",
      "tag_label": "ner",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        },
        "openai_transformer": {
            "type": "openai_transformer_byte_pair",
            "n_ctx": 50,
            "model_path": "allennlp/tests/fixtures/openai_transformer/transformer_small.tar.gz"
        },
      }
    },
    "train_data_path": "allennlp/tests/fixtures/data/conll2003.txt",
    "validation_data_path": "allennlp/tests/fixtures/data/conll2003.txt",
    "model": {
        "type": "crf_tagger",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "tokens": ["tokens"],
                "openai_transformer": ["openai_transformer", "openai_transformer-offsets"]
            },
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                },
                "openai_transformer": {
                    "type": "openai_transformer_embedder",
                    "transformer": {
                        "model_path": "allennlp/tests/fixtures/openai_transformer/transformer_small.tar.gz",
                        "embedding_dim": 10,
                        "num_heads": 2,
                        "num_layers": 2,
                        "vocab_size": 50,
                        "n_ctx": 50
                    }
                }
            }
        },
        "encoder": {
            "type": "gru",
            "input_size": 20,  // 10 + 10
            "hidden_size": 15,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": true
        },
        "regularizer": [
            ["transitions$", {"type": "l2", "alpha": 0.01}]
        ]
    },
    "iterator": {"type": "basic", "batch_size": 32},
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 1,
        "cuda_device": -1
    }
}

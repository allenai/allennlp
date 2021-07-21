{
    "steps": {
        "dataset": {
            "type": "dataset_reader_adapter",
            "reader": "sequence_tagging",
            "splits": {
              "train": "test_fixtures/data/sequence_tagging.tsv",
              "validation": "test_fixtures/data/sequence_tagging.tsv",
            }
        },
        "trained_model": {
            "type": "training",
            "dataset": "dataset",
            "training_split": "train",
            "data_loader": {
               "type": "sampler",
               "batch_sampler": {
                    "type": "bucket",
                    "sorting_keys": ["tokens"],
                    "padding_noise": 0.0,
                    "batch_size" : 80
               }
            },
            "model": {
                "type": "simple_tagger",
                "text_field_embedder": {
                  "token_embedders": {
                    "tokens": {
                        "type": "embedding",
                        "projection_dim": 2,
                        "pretrained_file": "test_fixtures/embeddings/glove.6B.100d.sample.txt.gz",
                        "embedding_dim": 100,
                        "trainable": true
                    }
                  }
                },
                "encoder": {
                  "type": "lstm",
                  "input_size": 2,
                  "hidden_size": 4,
                  "num_layers": 1
                }
            },
            "optimizer": {
              "type": "adam",
              "lr": 0.01
            },
            "grad_norm": 1.0,
            "num_epochs": 40,
            "patience": 500,
        },
        "evaluation": {
            "type": "evaluation",
            "dataset": "dataset",
            "model": {
                "type": "ref",
                "ref": "trained_model"
            }   # TODO: Figure out why this doesn't work as a string.
        }
    }
}

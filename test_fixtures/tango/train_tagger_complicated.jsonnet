{
    "steps": {
        "dataset": {
            "type": "dataset_reader_adapter",
            "reader": "sequence_tagging",
            "splits": {
              "train": "test_fixtures/data/sequence_tagging.tsv",
              "validation": "test_fixtures/data/sequence_tagging.tsv"
            }
        },
        "trained_model": {
            "type": "training",
            "dataset": { "ref": "dataset" },
            "training_split": "train",
            "validation_split": "validation",
            "data_loader": {
                "type": "batches_per_epoch",
                "inner": {
                    "batch_size": 1,
                },
                "batches_per_epoch": 20
            },
            "validation_data_loader": {
                "batch_size": 16
            },
            "no_grad": ['text_field_embedder.token_embedder_tokens._projection.weight'],
            "limit_batches_per_epoch": 10,
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
            "checkpointer": {
              "keep_most_recent_by_count": null
            }
        },
        "evaluation": {
            "type": "evaluation",
            "dataset": { "ref": "dataset" },
            "model": { "ref": "trained_model" },
            "data_loader": {
                "batch_size": 16
            },
        }
    }
}

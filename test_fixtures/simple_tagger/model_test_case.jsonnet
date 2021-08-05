{
  "dataset_reader":{"type":"sequence_tagging"},
  "train_data_path": "test_fixtures/data/sequence_tagging.tsv",
  "validation_data_path": "test_fixtures/data/sequence_tagging.tsv",
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
  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "padding_noise": 0.0,
        "batch_size" : 80
    }
  },
  "trainer": {
    "num_epochs": 40,
    "grad_norm": 1.0,
    "patience": 500,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}

{
  "dataset_reader": {
    "type": "simple_language_modeling",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "token_characters": {
        "type": "elmo_characters"
      }
    },
    "start_tokens": ["<S>"],
    "end_tokens": ["</S>"]
  },
  "train_data_path": "allennlp/tests/fixtures/language_model/sentences.txt",
  "validation_data_path": "allennlp/tests/fixtures/language_model/sentences.txt",
  "vocabulary": {
      "tokens_to_add": {
          "tokens": ["<S>", "</S>"],
          "token_characters": ["<>/S"]
      },
  },
  "model": {
    "type": "language_model",
    "bidirectional": true,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "token_embedders": {
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 4
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 4,
                "filters": [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                "num_highway": 2,
                "projection_dim": 16,
                "projection_location": "after_cnn"
            }
        }
      }
    },
    "contextualizer": {
        "type": "lstm",
        "bidirectional": true,
        "num_layers": 3,
        "input_size": 16,
        "hidden_size": 7,
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    },
    "log_batch_size_period": 1
  }
}

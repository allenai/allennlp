{
  "random_seed": 4536,
  "numpy_seed": 9834,
  "pytorch_seed": 953,
  "dataset_reader": {
    "type": "wikitables",
    "tables_directory": "/wikitables_tagged",
    "offline_logical_forms_directory": "/offline_search_output/",
    "max_offline_logical_forms": 60,
    "lazy": false
  },
  "validation_dataset_reader": {
    "type": "wikitables",
    "tables_directory": "/wikitables_tagged",
    "keep_if_no_logical_forms": true,
    "lazy": false
  },
  "vocabulary": {
    "min_count": {"tokens": 3},
    "tokens_to_add": {"tokens": ["-1"]}
  },
  "train_data_path": "/wikitables_raw_data/random-split-1-train.examples",
  "validation_data_path": "/wikitables_raw_data/random-split-1-dev.examples",
  "model": {
    "type": "wikitables_mml_parser",
    "question_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 200,
        "trainable": true
      }
    },
    "action_embedding_dim": 100,
    "encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 100,
      "bidirectional": true,
      "num_layers": 1
    },
    "entity_encoder": {
      "type": "boe",
      "embedding_dim": 200,
      "averaged": true
    },
    "decoder_beam_search": {
      "beam_size": 10
    },
    "max_decoding_steps": 16,
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "dropout": 0.5
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 1
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0,
    "grad_norm": 5.0,
    "validation_metric": "+denotation_acc",
    "optimizer": {
      "type": "sgd",
      "lr": 0.1
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.99
    }
  }
}

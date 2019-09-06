{
  "random_seed": 4536,
  "numpy_seed": 9834,
  "pytorch_seed": 953,
  "dataset_reader": {
    "type": "wikitables",
    "lazy": false,
    "output_agendas": true,
    "tables_directory": "/wikitables_tagged/",
    "keep_if_no_logical_forms": true
  },
  "vocabulary": {
    "min_count": {"tokens": 3},
    "tokens_to_add": {"tokens": ["-1"]}
  },
  "train_data_path": "/wikitables_raw_data/random-split-1-train.examples",
  "validation_data_path": "/wikitables_raw_data/random-split-1-dev.examples",
  "model": {
    "type": "wikitables_erm_parser",
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
    "checklist_cost_weight": 0.2,
    "max_decoding_steps": 18,
    "decoder_beam_size": 50,
    "decoder_num_finished_states": 100,
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "dropout": 0.5,
    "mml_model_file": "/mml_model/model.tar.gz"
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["question", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size" : 10
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 5,
    "validation_metric": "+denotation_acc",
    "cuda_device": -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}

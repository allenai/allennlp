// The Wikitables data is available at https://ppasupat.github.io/WikiTableQuestions/
{
  "dataset_reader": {
    "type": "wikitables",
    "lazy": false,
    "tables_directory": "/wikitables/",
    "dpd_output_directory": "/wikitables/dpd_output/",
    "question_token_indexers": {
      "tokens": {"type": "single_id"}
    }
  },
  "validation_dataset_reader": {
    "type": "wikitables",
    "lazy": false,
    "tables_directory": "/wikitables/",
    "dpd_output_directory": "/wikitables/dpd_output/",
    "question_token_indexers": {
      "tokens": {"type": "single_id"}
    },
    "keep_if_no_dpd": true
  },
  "vocabulary": {
    "min_count": {"tokens": 3},
    "tokens_to_add": {"tokens": ["-1", "0", "1"]}
  },
  "train_data_path": "/wikitables_preprocessed/train.jsonl",
  "validation_data_path": "/wikitables_preprocessed/validation.jsonl",
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
    "training_beam_size": 5,
    "max_decoding_steps": 40,
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
    "num_epochs": 20,
    "patience": 5,
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

local pretrained = function(module_path, frozen=false) {"_pretrained": {
    "archive_file": std.extVar("ARCHIVE_PATH"),
    "module_path": module_path,
    "freeze": frozen
}};

{
  "dataset_reader": {
    "type": "text_classification_json",
    "tokenizer": {
      "type": "whitespace"
    }
  },
  
  "train_data_path": "allennlp/tests/fixtures/data/movies_train.jsonl",
  "validation_data_path": "allennlp/tests/fixtures/data/movies_train.jsonl",
  "vocabulary": {
    "type": "extend",
    "directory": "/tmp/taskA/vocabulary"
  },
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": pretrained("_text_field_embedder"),
    "seq2seq_encoder": pretrained("_encoder"),
    "seq2vec_encoder": {
        "type": "boe",
        "embedding_dim": 200
    },
    "feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": [200, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.15, 0.0]
    },
  },
  "data_loader": {
    "type": "default",
    "batch_size": 10
  },
  "trainer": {
    "num_epochs": 1,
    "patience": 5,
    "cuda_device": -1,
    "grad_norm": 40,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 5e-3
    },
  }
}

local config = import "experiment_unsampled.jsonnet";

config + {
  "model"+: {
    "num_samples": 10,
    "sparse_embeddings": true,
    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": 16,
        "hidden_dim": 7,
        "num_layers": 3,
        "dropout": 0.1,
        "input_dropout": 0.1
    }
  }
}
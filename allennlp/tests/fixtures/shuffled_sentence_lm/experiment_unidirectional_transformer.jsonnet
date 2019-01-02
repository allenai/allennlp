local config = import "experiment_unidirectional_unsampled.jsonnet";

config + {
  "model"+: {
    "num_samples": 10,
    "sparse_embeddings": true,
    "contextualizer": {
        "type": "stacked_self_attention",
        "input_dim": 16,
        "hidden_dim": 20,
        "projection_dim": 6,
        "feedforward_hidden_dim": 5,
        "num_attention_heads": 3,
        "num_layers": 3,
        "dropout_prob": 0.1
    }
  }
}

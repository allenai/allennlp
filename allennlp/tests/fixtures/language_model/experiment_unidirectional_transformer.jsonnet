local config = import "experiment_unidirectional_unsampled.jsonnet";

config + {
  "model"+: {
    "num_samples": 10,
    "sparse_embeddings": true,
    "contextualizer": {
        "type": "pytorch_transformer",
        "input_dim": 16,
        "feedforward_hidden_dim": 20,
        "num_attention_heads": 4,
        "num_layers": 3,
        "dropout_prob": 0.1,
        "positional_encoding": "sinusoidal"
    }
  }
}

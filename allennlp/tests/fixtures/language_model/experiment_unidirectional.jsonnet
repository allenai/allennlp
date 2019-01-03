local config = import "experiment_unidirectional_unsampled.jsonnet";

config + {
  "model"+: {
    "num_samples": 10,
    "sparse_embeddings": true
  }
}

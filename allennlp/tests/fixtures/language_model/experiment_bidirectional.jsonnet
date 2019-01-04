local config = import "experiment_bidirectional_unsampled.jsonnet";

config + {
  "model"+: {
    "num_samples": 10,
    "sparse_embeddings": true
  }
}

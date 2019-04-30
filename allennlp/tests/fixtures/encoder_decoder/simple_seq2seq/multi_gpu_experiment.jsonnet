local config = import "experiment.json";

config + {
  "model"+: {
    "encoder": {
        "type": "stacked_self_attention",
        "input_dim": 47,
        "hidden_dim": 10,
        "projection_dim": 10,
        "feedforward_hidden_dim": 47,
        "num_attention_heads": 2,
        "num_layers": 2,
        "dropout_prob": 0.1
    }
  },
  "trainer"+: {
    "cuda_device": [0]
  }
}

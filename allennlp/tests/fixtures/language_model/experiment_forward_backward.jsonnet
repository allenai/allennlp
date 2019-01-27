local config = import "experiment_unsampled.jsonnet";

config + {
  "model"+: {
    contextualizer :: super.contextualizer,
    "forward_contextualizer": {
        "type": "lstm",
        "input_size": 16,
        "hidden_size": 7,
        "num_layers": 3,
        "dropout": 0.1
    },
    "backward_contextualizer": {
        "type": "gru",
        "input_size": 16,
        "hidden_size": 7,
        "num_layers": 3,
        "dropout": 0.1
    }
  }
}
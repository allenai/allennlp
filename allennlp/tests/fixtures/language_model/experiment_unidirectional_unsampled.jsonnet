local config = import "experiment_unsampled.jsonnet";

config + {
    "model"+: {
        "bidirectional": false,
        "contextualizer" +: {
            "bidirectional": false
        }
    }
}

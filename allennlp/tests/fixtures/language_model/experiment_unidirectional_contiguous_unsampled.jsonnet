local config = import "experiment_unsampled.jsonnet";

config + {
    "dataset_reader"+: {
        "type": "language_modeling",
        "batch_size": 2,
    },
    "model"+: {
        "bidirectional": false,
        "contextualizer" +: {
            "bidirectional": false,
            // This is necessary for contiguous text LMs
            "stateful": true
        }
    },
    "iterator"+: {
        "type": "language_modeling",
        batch_size :: super.batch_size
    }
}

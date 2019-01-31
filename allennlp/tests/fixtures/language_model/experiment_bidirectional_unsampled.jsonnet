local config = import "experiment_unsampled.jsonnet";

config + {
    "model"+: {
        "type": "bidirectional_language_model",
        // Hide the bidirectional field, since the bidirectional_language_model
        // does not accept it.
        bidirectional:: super.bidirectional,
    }
}

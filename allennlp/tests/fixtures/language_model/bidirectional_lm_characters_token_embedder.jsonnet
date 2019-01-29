local config = import "characters_token_embedder.json";

config + {
  "model"+: {
    "text_field_embedder"+: {
      "token_embedders"+: {
        "elmo"+: {
          "type": "bidirectional_lm_token_embedder",
        }
      }
    }
  }
}

local config = import "bidirectional_lm_characters_token_embedder.jsonnet";

config + {
  "model"+: {
    "text_field_embedder"+: {
      "token_embedders"+: {
        "elmo"+: {
          "bos_eos_tokens": null,
          "remove_bos_eos": false
        }
      }
    }
  }
}

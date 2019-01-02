local config = import "characters_token_embedder.json";

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

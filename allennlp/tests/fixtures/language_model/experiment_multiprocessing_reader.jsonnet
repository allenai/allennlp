local config = import "experiment_unsampled.jsonnet";

config + {
  "dataset_reader": {
    "type": "multiprocess",
    "base_reader": {
      "type": "simple_language_modeling",
      "tokenizer": {
        "type": "word",
        "word_splitter": {
          "type": "just_spaces"
        }
      },
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        },
        "token_characters": {
          "type": "elmo_characters"
        }
      },
      "start_tokens": ["<S>"],
      "end_tokens": ["</S>"]
    },
    "num_workers": 1,
    "output_queue_size": 1000
  },

  // Note the glob on the end of these paths.
  "train_data_path": "allennlp/tests/fixtures/language_model/sentences*",
  "validation_data_path": "allennlp/tests/fixtures/language_model/sentences*",
  "test_data_path": "allennlp/tests/fixtures/language_model/sentences*",
}

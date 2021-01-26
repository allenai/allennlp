local transformer_model = "bert-base-uncased";
local transformer_dim = 786;

{
  "dataset_reader": {
    "type": "snli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 128
      }
    }
  },
  "train_data_path": "data/multinli_1.0_train_first10000.jsonl",
  "validation_data_path": "data/multinli_1.0_dev_matched.jsonl",
  "test_data_path": "data/multinli_1.0_dev_mismatched.jsonl",
  "interpreter": {
    "type": "simple-influence",
    "k": 20,
  }
}

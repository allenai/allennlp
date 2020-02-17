// Configuration for a RoBERTa sentiment analysis classifier, using the binary Stanford Sentiment Treebank (Socher at al. 2013).

local transformer_model = "roberta-large";
local transformer_dim = 1024;
local cls_is_last_token = false;

{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model
      }
    }
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model
      }
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "dropout": 0.1
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "tokens___token_ids"]],
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 10,
      "num_steps_per_epoch": 3088,
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}

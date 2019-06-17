{
  "dataset_reader": {
    "type": "quora_paraphrase",
    "lazy": false,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/quora-question-paraphrase/train.tsv",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/quora-question-paraphrase/dev.tsv",
  "model": {
    "type": "bimpm",
    "dropout": 0.1,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
        "embedding_dim": 300,
        "trainable": false,
        "padding_index": 0
      },
      "token_characters": {
        "type": "character_encoding",
        "embedding": {
          "embedding_dim": 20,
          "padding_index": 0
        },
        "encoder": {
          "type": "gru",
          "input_size": 20,
          "hidden_size": 50,
          "num_layers": 1,
          "bidirectional": true
        }
      }
    },
    "matcher_word": {
      "is_forward": true,
      "hidden_dim": 400,
      "num_perspectives": 10,
      "with_full_match": false
    },
    "encoder1": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 400,
      "hidden_size": 200,
      "num_layers": 1
    },
    "matcher_forward1": {
      "is_forward": true,
      "hidden_dim": 200,
      "num_perspectives": 10
    },
    "matcher_backward1": {
      "is_forward": false,
      "hidden_dim": 200,
      "num_perspectives": 10
    },
    "encoder2": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 400,
      "hidden_size": 200,
      "num_layers": 1
    },
    "matcher_forward2": {
      "is_forward": true,
      "hidden_dim": 200,
      "num_perspectives": 10
    },
    "matcher_backward2": {
      "is_forward": false,
      "hidden_dim": 200,
      "num_perspectives": 10
    },
    "aggregator":{
      "type": "lstm",
      "bidirectional": true,
      "input_size": 264,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.1
    },
    "classifier_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.1, 0.0]
    },
    "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*linear_layers.*bias", {"type": "constant", "val": 0}],
      [".*weight_ih.*", {"type": "xavier_normal"}],
      [".*weight_hh.*", {"type": "orthogonal"}],
      [".*bias.*", {"type": "constant", "val": 0}],
      [".*matcher.*match_weights.*", {"type": "kaiming_normal"}]
    ]
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.1,
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.0005
    }
  }
}

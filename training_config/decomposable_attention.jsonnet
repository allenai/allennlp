// Configuraiton for a textual entailment model based on:
//  Parikh, Ankur P. et al. “A Decomposable Attention Model for Natural Language Inference.” EMNLP (2016).
{
  "dataset_reader": {
    "type": "snli",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
      "end_tokens": ["@@NULL@@"]
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
  "model": {
    "type": "decomposable_attention",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "projection_dim": 200,
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "attend_feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": 64
  },

  "trainer": {
    "num_epochs": 140,
    "patience": 20,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}

local train_size = 200;
local batch_size = 8;
local gradient_accumulation_batch_size = 2;
local num_epochs = 20;
local learning_rate = 1e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "bert-base-uncased";
local cuda_device = -1;

{
  "dataset_reader": {
    "type": "base_esim_reader",
     "token_indexers": {
        "tokens": {
            "type": "single_id",
        }
     },
     "sample": -1
  },
  "validation_dataset_reader": {
    "type": "base_esim_reader",
    "token_indexers": {
        "tokens": {
            "type": "single_id",
        }
     },
    "sample": -1
     //"num_choices": "[NUM_OF_CHOICES]"
  },
  "train_data_path": "https://olmpics.s3.us-east-2.amazonaws.com/challenge/negation/negation_unfilt_not_to_definitely_triplets_7132_train.jsonl.gz",
  "validation_data_path": "https://olmpics.s3.us-east-2.amazonaws.com/challenge/negation/negation_unfilt_not_to_definitely_triplets_7132_dev.jsonl.gz",

  "model": {
    "type": "esim_baseline",
    "dropout": 0.3,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        //"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
        "embedding_dim": 50,
        "trainable": true
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 50,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "similarity_function": {"type": "dot_product"},
    "projection_feedforward": {
      "input_dim": 2400,
      "hidden_dims": 300,
      "num_layers": 1,
      "activations": "relu"
    },
    "inference_encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "output_feedforward": {
      "input_dim": 2400,
      "num_layers": 1,
      "hidden_dims": 300,
      "activations": "relu",
      "dropout": 0.3
    },
    "output_logit": {
      "input_dim": 300,
      "num_layers": 1,
      "hidden_dims": 1,
      "activations": "linear"
    },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_uniform"}],
      [".*linear_layers.*bias", {"type": "constant", "val": 0}],
//      [".*weight_ih.*", {"type": "xavier_uniform"}], these should get initialized already!
//      [".*weight_hh.*", {"type": "orthogonal"}],
//      [".*bias_ih.*", {"type": "constant", "val": 0}],
//      [".*bias_hh.*", {"type": "constant", "val": 1}]
     ]
   },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"]],
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.0004
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 0,
    "num_epochs": 75,
    "grad_norm": 10.0,
    "patience": 10,
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 0
    }
  }
}
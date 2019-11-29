local train_size = 200;
local batch_size = 8;
local gradient_accumulation_batch_size = 2;
local num_epochs = 20;
local learning_rate = 3e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "bert-base-uncased";
local cuda_device = -1;

{
  "dataset_reader": {
    "type": "base_mlm_reader",
     "token_indexers": {
        "tokens": {
            "type": "single_id",
        }
     },
     "sample": -1
  },
  "validation_dataset_reader": {
    "type": "base_mlm_reader",
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
    "type": "mlm_baseline",
    "dropout": 0.3,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
        "embedding_dim": 50,
        "trainable": false
      }
    }
   },
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.0004,
        "weight_decay" : weight_decay
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
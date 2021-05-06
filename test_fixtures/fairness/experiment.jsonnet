local transformer_model = "epwalsh/bert-xsmall-dummy";
local transformer_dim = 20;

{
  "dataset_reader":{
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
        "max_length": 512
      }
    }
  },
  "train_data_path": "test_fixtures/fairness/snli_train.jsonl",
  "validation_data_path": "test_fixtures/fairness/snli_dev.jsonl",
  "test_data_path": "test_fixtures/fairness/snli_test.jsonl",
  "model": {
    "type": "allennlp.fairness.bias_mitigator_applicator.BiasMitigatorApplicator", 
    "base_model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "max_length": 512
          }
        }
      },
      "seq2vec_encoder": {
        "type": "cls_pooler",
        "embedding_dim": transformer_dim,
      },
      "feedforward": {
        "input_dim": transformer_dim,
        "num_layers": 1,
        "hidden_dims": transformer_dim,
        "activations": "tanh"
      },
      "dropout": 0.1,
      "namespace": "tags"
    },
    "bias_mitigator": {
      "type": "hard",
      "bias_direction": {
        "type": "paired_pca",
        "seed_word_pairs_file": "test_fixtures/fairness/definitional_pairs.json",
        "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      },
      "equalize_word_pairs_file": "test_fixtures/fairness/equalize_pairs.json",
      "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "padding_noise": 0.0,
      "batch_size" : 80
    }
  },
  "trainer": {
    "num_epochs": 5,
    "grad_norm": 1.0,
    "patience": 500,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}

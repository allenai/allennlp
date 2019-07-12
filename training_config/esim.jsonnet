{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
    "model": {
        "type": "esim",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "similarity_function": {
            "type": "dot_product"
        },
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
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": 300,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}

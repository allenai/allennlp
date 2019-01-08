{
    "dataset_reader": {
        "type": "textcat",
        "segment_sentences": true,
        "debug": true,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
    },
  "datasets_for_vocab_creation": ["train"],
"train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/train.jsonl",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/dev.jsonl",
    "model": {
        "type": "han",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        },
        "word_encoder": {
           "type": "gru",
           "num_layers": 1,
           "bidirectional": true,
	       "input_size": 300,
           "hidden_size": 50,
        },
         "sentence_encoder": {
           "type": "gru",
           "num_layers": 1,
           "bidirectional": true,
	       "input_size": 100,
           "hidden_size": 50,
        },
        "word_attention": {
            "type": "attention_encoder",
            "input_dim": 100,
            "context_vector_dim": 100
        },
        "sentence_attention": {
            "type": "attention_encoder",
            "input_dim": 100,
            "context_vector_dim": 100
        },
        "classification_layer": {
            "input_dim": 100,
            "num_layers": 1,
            "hidden_dims": 4,
            "dropout": 0.2457355626352195,
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
        "sorting_keys": [["tokens", "list_num_tokens"]],
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

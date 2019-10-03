{
    "dataset_reader": {
        "type": "quac",
        "lazy": true,
        "num_context_answers": 2,
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
            "token_characters": {
                "type": "characters",
                "character_tokenizer": {
                    "byte_encoding": "utf-8",
                    "end_tokens": [
                        260
                    ],
                    "start_tokens": [
                        259
                    ]
                },
                "min_padding_length": 5
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 10,
        "max_instances_in_memory": 1000,
        "sorting_keys": [
            [
                "question",
                "num_fields"
            ],
            [
                "passage",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "dialog_qa",
        "dropout": 0.2,
        "initializer": [],
        "marker_embedding_dim": 10,
        "num_context_answers": 2,
        "phrase_layer": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 1144,  // elmo (1024) + cnn (100) + num_context_answers (2) * marker_embedding_dim (10)
            "num_layers": 1
        },
        "residual_encoder": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 200,
            "num_layers": 1
        },
        "span_end_encoder": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 400,
            "num_layers": 1
        },
        "span_start_encoder": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 200,
            "num_layers": 1
        },
        "text_field_embedder": {
            "elmo": {
                "type": "elmo_token_embedder",
                "do_layer_norm": false,
                "dropout": 0.2,
                "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            },
            "token_characters": {
                "type": "character_encoding",
                "dropout": 0.2,
                "embedding": {
                    "embedding_dim": 20,
                    "num_embeddings": 262
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 20,
                    "ngram_filter_sizes": [
                        5
                    ],
                    "num_filters": 100
                }
            }
        }
    },
    "train_data_path": "https://s3.amazonaws.com/my89public/quac/train_5000.json",
    "validation_data_path": "https://s3.amazonaws.com/my89public/quac/val.json",
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 3
        },
        "num_epochs": 30,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
        "patience": 10,
        "validation_metric": "+f1"
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": 3,
        "max_instances_in_memory": 1000,
        "sorting_keys": [
            [
                "question",
                "num_fields"
            ],
            [
                "passage",
                "num_tokens"
            ]
        ]
    }
}

// Configuration for the basic QANet model from "QANet: Combining Local
// Convolution with Global Self-Attention for Reading Comprehension"
// (https://arxiv.org/abs/1804.09541).
{
    "dataset_reader": {
        "type": "squad",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 5
            }
        },
        "passage_length_limit": 400,
        "question_length_limit": 50,
        "skip_invalid_examples": true
    },
    "validation_dataset_reader": {
        "type": "squad",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 5
            }
        },
        "passage_length_limit": 1000,
        "question_length_limit": 100,
        "skip_invalid_examples": false
    },
    "vocabulary": {
        "min_count": {
            "token_characters": 200
        },
        "pretrained_files": {
            // This embedding file is created from the Glove 840B 300d embedding file.
            // We kept all the original lowercased words and their embeddings. But there are also many words
            // with only the uppercased version. To include as many words as possible, we lowered those words
            // and used the embeddings of uppercased words as an alternative.
            "tokens": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.lower.converted.zip"
        },
        "only_include_pretrained_words": true
    },
    "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/squad/squad-train-v1.1.json",
    "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json",
    "model": {
        "type": "qanet",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.lower.converted.zip",
                    "embedding_dim": 300,
                    "trainable": false
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 64
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 64,
                        "num_filters": 200,
                        "ngram_filter_sizes": [
                            5
                        ]
                    }
                }
            }
        },
        "num_highway_layers": 2,
        "phrase_layer": {
            "type": "qanet_encoder",
            "input_dim": 128,
            "hidden_dim": 128,
            "attention_projection_dim": 128,
            "feedforward_hidden_dim": 128,
            "num_blocks": 1,
            "num_convs_per_block": 4,
            "conv_kernel_size": 7,
            "num_attention_heads": 8,
            "dropout_prob": 0.1,
            "layer_dropout_undecayed_prob": 0.1,
            "attention_dropout_prob": 0
        },
        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 128,
            "tensor_2_dim": 128,
            "combination": "x,y,x*y"
        },
        "modeling_layer": {
            "type": "qanet_encoder",
            "input_dim": 128,
            "hidden_dim": 128,
            "attention_projection_dim": 128,
            "feedforward_hidden_dim": 128,
            "num_blocks": 7,
            "num_convs_per_block": 2,
            "conv_kernel_size": 5,
            "num_attention_heads": 8,
            "dropout_prob": 0.1,
            "layer_dropout_undecayed_prob": 0.1,
            "attention_dropout_prob": 0
        },
        "dropout_prob": 0.1,
        "regularizer": [
            [
                ".*",
                {
                    "type": "l2",
                    "alpha": 1e-07
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "passage",
                "num_tokens"
            ],
            [
                "question",
                "num_tokens"
            ]
        ],
        "batch_size": 25,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs": 50,
        "grad_norm": 5,
        "patience": 10,
        "validation_metric": "+em",
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-07
        },
        "moving_average": {
            "type": "exponential",
            "decay": 0.9999
        }
    }
}
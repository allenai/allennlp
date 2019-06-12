// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
{
    "dataset_reader": {
        "type": "universal_dependencies_multilang",
        "languages": ["en", "de", "it", "fr", "pt", "sv"],
        "alternate": true,
        "instances_per_file": 32,
        "is_first_pass_for_vocab": true,
        "lazy": true,
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        },
        "use_language_specific_pos": false
    },
    "iterator": {
        "type": "same_language",
        "batch_size": 32,
        "sorting_keys": [["words", "num_tokens"]],
        "instances_per_epoch": 32000
    },
    "model": {
        "type": "biaffine_parser_multilang",
        "arc_representation_dim": 500,
        "dropout": 0.33,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.33,
            "hidden_size": 200,
            "input_size": 1074,
            "num_layers": 3
        },
        "langs_for_early_stop": [
            "en",
            "de",
            "it",
            "fr",
            "pt",
            "sv"
        ],
        "pos_tag_embedding": {
            "embedding_dim": 50,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder_multilang",
                    "aligning_files": {
                        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_best_mapping.pth",
                        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/es_best_mapping.pth",
                        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_best_mapping.pth",
                        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/it_best_mapping.pth",
                        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/pt_best_mapping.pth",
                        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/sv_best_mapping.pth",
                        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_best_mapping.pth"
                    },
                    "do_layer_norm": false,
                    "dropout": 0.3,
                    "scalar_mix_parameters": [
                        -9e10,
                        1,
                        -9e10
                    ],
                    "options_files": {
                        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json"
                    },
                    "weight_files": {
                        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_weights.hdf5",
                        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/es_weights.hdf5",
                        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_weights.hdf5",
                        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/it_weights.hdf5",
                        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/pt_weights.hdf5",
                        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/sv_weights.hdf5",
                        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_weights.hdf5"
                    }
                }
            }
        }
    },
    // UDTB v2.0 is available at https://github.com/ryanmcd/uni-dep-tb
    // Set TRAIN_PATHNAME='std/**/*train.conll'
    "train_data_path": std.extVar("TRAIN_PATHNAME"),
    "validation_data_path": std.extVar("DEV_PATHNAME"),
    "test_data_path": std.extVar("TEST_PATHNAME"),
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 40,
        "optimizer": "adam",
        "patience": 10,
        "validation_metric": "+LAS_AVG"
    },
    "validation_dataset_reader": {
        "type": "universal_dependencies_multilang",
        "languages": ["en", "es", "de", "it", "fr", "pt", "sv"],
        "alternate": false,
        "lazy": true,
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        },
        "use_language_specific_pos": false
    },
    "validation_iterator": {
        "type": "same_language",
        "sorting_keys": [["words", "num_tokens"]],
        "batch_size": 32
    }
}

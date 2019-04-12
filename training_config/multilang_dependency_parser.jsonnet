// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://arxiv.org/abs/1902.09492 (NAACL 2019)
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
        "type": "same_lang",
        "batch_size": 32,
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
                        "en": "https://www.dropbox.com/s/nufj4pxxgv5838r/en_best_mapping.pth?dl=1",
                        "es": "https://www.dropbox.com/s/6kqot8ssy66d5u0/es_best_mapping.pth?dl=1",
                        "fr": "https://www.dropbox.com/s/0zdlanjhajlgflm/fr_best_mapping.pth?dl=1",
                        "it": "https://www.dropbox.com/s/gg985snnhajhm5i/it_best_mapping.pth?dl=1",
                        "pt": "https://www.dropbox.com/s/skdfz6zfud24iup/pt_best_mapping.pth?dl=1",
                        "sv": "https://www.dropbox.com/s/o7v64hciyifvs8k/sv_best_mapping.pth?dl=1",
                        "de": "https://www.dropbox.com/s/u9cg19o81lpm0h0/de_best_mapping.pth?dl=1"
                    },
                    "do_layer_norm": false,
                    "dropout": 0.3,
                    "scalar_mix_parameters": [
                        -9e10,
                        1,
                        -9e10
                    ],
                    "options_files": {
                        "en": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1",
                        "es": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1",
                        "fr": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1",
                        "it": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1",
                        "pt": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1",
                        "sv": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1",
                        "de": "https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=1"
                    },
                    "weight_files": {
                        "en": "https://www.dropbox.com/s/1h62kc1qdcuyy2u/en_weights.hdf5?dl=1",
                        "es": "https://www.dropbox.com/s/ygfjm7zmufl5gu2/es_weights.hdf5?dl=1",
                        "fr": "https://www.dropbox.com/s/mm64goxb8wbawhj/fr_weights.hdf5?dl=1",
                        "it": "https://www.dropbox.com/s/owfou7coi04dyxf/it_weights.hdf5?dl=1",
                        "pt": "https://www.dropbox.com/s/ul82jsal1khfw5b/pt_weights.hdf5?dl=1",
                        "sv": "https://www.dropbox.com/s/boptz21zrs4h3nw/sv_weights.hdf5?dl=1",
                        "de": "https://www.dropbox.com/s/2kbjnvb12htgqk8/de_weights.hdf5?dl=1"
                    }
                }
            }
        }
    },
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
        "type": "same_lang",
        "batch_size": 32
    }
}

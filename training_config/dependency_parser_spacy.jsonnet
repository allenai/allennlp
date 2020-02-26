{
    "dataset_reader":{
      "type":"universal_dependencies",
      "token_indexers": {
        "tokens": {
          "type": "spacy"
        },
      },

      "tokenizer": {
        "type": "word",
        "word_splitter": {
          "type": "spacy",
          "split_on_spaces": true,
          "keep_spacy_tokens": true,
          "pos_tags": true,
        },
      },
    },
    "train_data_path": "/Users/markn/allen_ai/data/dependency-parsing/en-ud/en_ewt-ud-train.conllu",
    "validation_data_path": "/Users/markn/allen_ai/data/dependency-parsing/en-ud/en_ewt-ud-dev.conllu",
    "model": {
      "type": "biaffine_parser",
      "text_field_embedder": {
        "tokens": {
          "type": "pass_through",
          "hidden_dim": 96,
        }
      },
//      "pos_tag_embedding":{
//        "embedding_dim": 100,
//        "vocab_namespace": "pos",
//        "sparse": true
//      },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 96,
        "hidden_size": 100,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.3,
        "use_highway": true
      },
      "use_mst_decoding_for_validation": false,
      "arc_representation_dim": 200,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
      "initializer": [
        [".*projection.*weight", {"type": "xavier_uniform"}],
        [".*projection.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]]
    },

    "iterator": {
      "type": "bucket",
      "sorting_keys": [["words", "num_tokens"]],
      "batch_size" : 16
    },
    "trainer": {
      "num_epochs": 50,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": -1,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
  }


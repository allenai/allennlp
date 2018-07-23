{
    "dataset_reader": {
      "type": "conll2003",
      "tag_label": "ner",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        },
        "openai_transformer": {
            "type": "openai_transformer_byte_pair",
            "transformer_model_path": "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
        },
      }
    },
    "train_data_path": "allennlp/tests/fixtures/data/conll2003.txt",
    "validation_data_path": "allennlp/tests/fixtures/data/conll2003.txt",
    "model": {
      "type": "crf_tagger",
      "text_field_embedder": {
              "tokens": {
                "type": "embedding",
                "embedding_dim": 50
              },
              "openai_transformer-text_tokens": {
                  "type": "openai_transformer_embedder",
                  "transformer_model_path": "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
              }
      },
      "encoder": {
              "type": "gru",
              "input_size": 818,  // 50 + 768
              "hidden_size": 25,
              "num_layers": 2,
              "dropout": 0.5,
              "bidirectional": true
      },
      "regularizer": [
        ["transitions$", {"type": "l2", "alpha": 0.01}]
      ]
    },
    "iterator": {"type": "basic", "batch_size": 32},
    "trainer": {
      "optimizer": "adam",
      "num_epochs": 5,
      "cuda_device": -1
    }
  }

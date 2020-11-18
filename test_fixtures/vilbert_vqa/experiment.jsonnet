local model_name = "epwalsh/bert-xsmall-dummy";

{
  "dataset_reader": {
    "type": "vqav2",
    "image_dir": "test_fixtures/data/vqav2/images",
    "image_loader": "detectron",
    "image_featurizer": "null",
    "region_detector": "random",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name
      }
    }
  },
  "train_data_path": "unittest",
  "validation_data_path": "unittest",
  "vocabulary": {"min_count": {"answers": 2}},
  "datasets_for_vocab_creation": ["train"],
  "model": {
    "type": "vqa_vilbert",
    "text_embeddings": {
      "vocab_size": 250,
      "hidden_size": 20,
      "pad_token_id": 0,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "dropout": 0.0
    },
    "image_embeddings": {
      "feature_dim": 10,
      "hidden_dim": 200
    },
    "encoder": {
      # text
      "hidden_size1": 20,
      "num_hidden_layers1": 1,
      "intermediate_size1": 40,
      "num_attention_heads1": 1,
      "attention_dropout1": 0.1,
      "hidden_dropout1": 0.1,
      "biattention_id1": [0, 1],
      "fixed_layer1": 0,

      # vision
      "hidden_size2": 200,
      "num_hidden_layers2": 1,
      "intermediate_size2": 50,
      "num_attention_heads2": 1,
      "attention_dropout2": 0.0,
      "hidden_dropout2": 0.0,
      "biattention_id2": [0, 1],
      "fixed_layer2": 0,

      "combined_num_attention_heads": 2,
      "combined_hidden_size": 200,
      "activation": "gelu",
    },
    "pooled_output_dim": 100,
    "fusion_method": "sum",
  },
  "data_loader": {
    "batch_size": 4
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 0.00005
    },
    "num_epochs": 1,
  },
}

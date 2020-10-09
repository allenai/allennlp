local model_name = "epwalsh/bert-xsmall-dummy";
{
  "dataset_reader": {
    "type": "nlvr2",
    "image_dir": "test_fixtures/data/nlvr2",
    "image_loader": "detectron",
    "image_featurizer": "null",
    "region_detector": "null",
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
  "train_data_path": "test_fixtures/data/nlvr2/tiny-dev.json",
  "validation_data_path": "test_fixtures/data/nlvr2/tiny-dev.json",
  "model": {
    "type": "nlvr2_vilbert_from_pretrained",
    "pretrained_module": model_name,
    "image_feature_dim": 10,
    "image_hidden_size": 200,
    "image_num_hidden_layers": 1,
    "combined_hidden_size": 200,
    "pooled_output_dim": 100,
    "image_intermediate_size": 50,
    "image_attention_dropout": 0.0,
    "image_hidden_dropout": 0.0,
    "v_biattention_id": [0, 1],
    "t_biattention_id": [0, 1],
    "fixed_t_layer": 0,
    "fixed_v_layer": 0,
    "fusion_method": "sum",
    "pooled_dropout": 0.0,
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.05
    },
    "validation_metric": "+denotation_acc",
    "num_epochs": 1
  }
}

local model_name = "bert-base-uncased";
{
  "dataset_reader": {
    "type": "nlvr2_lxmert",
    "text_path_prefix": "test_fixtures/data/nlvr2/instances",
    "visual_path_prefix": "test_fixtures/data/nlvr2/object_features",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name
      }
    },
    "topk_images": -1
  },
  "train_data_path": "train",
  "validation_data_path": "train",
  "model": {
    "type": "nlvr2_vilbert_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 2048,
    "image_hidden_size": 24,
    "image_num_hidden_layers": 12,
    "combined_hidden_size": 36,
    "pooled_output_dim": 3,
    "image_intermediate_size": 7,
    "image_attention_dropout": 0.0,
    "image_hidden_dropout": 0.0,
    "v_biattention_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "t_biattention_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "fixed_t_layer": 0,
    "fixed_v_layer": 0,
    "fusion_method": "sum"
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 0.00005
    },
    "validation_metric": "+denotation_acc",
    "num_epochs": 1
  }
}

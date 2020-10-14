local model_name = "bert-base-uncased";
local effective_batch_size = 128;
local gpu_batch_size = 32;
local num_gpus = 1;

{
  "dataset_reader": {
    "type": "vqav2",
    #"image_dir": "/net/nfs2.corp/prior/datasets/coco",
    #"feature_cache_dir": "/net/nfs2.corp/prior/datasets/coco/coco_experiment_cache",
    #"image_dir": "/net/s3/allennlp/dirkg/data/vision/coco",
    #"feature_cache_dir": "/net/s3/allennlp/dirkg/data/vision/coco/feature_cache/vqa",
    "image_dir": "/Users/dirkg/Documents/data/vision/coco",
    "feature_cache_dir": "/Users/dirkg/Documents/data/vision/coco/feature_cache/vqa",
    "image_loader": "detectron",
    "image_featurizer": "resnet_backbone",
    "region_detector": "faster_rcnn",
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
    #"max_instances": 1000,
    "image_processing_batch_size": 16
  },
  "vocabulary": {"min_count": {"answers": 9}},
  "train_data_path": "balanced_real_train",
  "validation_data_path": "balanced_real_val",
  "model": {
    "type": "vqa_vilbert_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 2048,
    "image_hidden_size": 1024,
    "image_num_attention_heads": 8,
    "image_num_hidden_layers": 6,
    "combined_hidden_size": 1024,
    "combined_num_attention_heads": 8,
    "pooled_output_dim": 1024,
    "image_intermediate_size": 1024,
    "image_attention_dropout": 0.1,
    "image_hidden_dropout": 0.1,
    "v_biattention_id": [0, 1, 2, 3, 4, 5],
    "t_biattention_id": [6, 7, 8, 9, 10, 11],
    "fixed_t_layer": 0,
    "fixed_v_layer": 0,
    "fusion_method": "mul"
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true
  },
  [if num_gpus > 1 then "distributed"]: {
    #"cuda_devices": std.range(0, num_gpus - 1)
    "cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  "trainer": {
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 4e-5
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 300000 / 30
    },
    "validation_metric": "+fscore",
    "num_epochs": 20,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus)
  },
}

local model_name = "bert-base-uncased";
local effective_batch_size = 128;
local gpu_batch_size = 128;
local num_gpus = 1;

local vocabulary = {
  "type": "from_files",
  "directory": "https://storage.googleapis.com/allennlp-public-data/vqav2/vqav2_vocab.tar.gz"
};

{
  "dataset_reader": {
    "type": "vqav2",
    #"image_dir": "/mnt/tank/dirkg/data/vision/coco",
    #"feature_cache_dir": "/mnt/tank/dirkg/data/vision/vqa_feature_cache",
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
    },
    #"max_instances": 1000,
    "image_processing_batch_size": 32,
    "answer_vocab": vocabulary,
    "keep_unanswerable_questions": false
  },
  "validation_dataset_reader": self.dataset_reader {
    "keep_unanswerable_questions": true
  },
  "vocabulary": vocabulary,
  "train_data_path": ["balanced_real_train", "balanced_real_val[1000:]"],
  "validation_data_path": "balanced_real_val[:1000]",
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
    "image_biattention_id": [0, 1, 2, 3, 4, 5],
    "text_biattention_id": [6, 7, 8, 9, 10, 11],
    "text_fixed_layer": 0,
    "image_fixed_layer": 0,
    "fusion_method": "mul"
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    "max_instances_in_memory": 1024
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
      "warmup_steps": 300000 / 30,
      "num_steps_per_epoch": std.ceil(644401 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"])
    },
    "validation_metric": "+fscore",
    "num_epochs": 20,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus)
  },
}

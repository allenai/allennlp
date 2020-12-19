local model_name = "bert-base-uncased";
local effective_batch_size = 128;
local gpu_batch_size = 32;
local num_gpus = 0;

local datadir = "/net/s3/allennlp/akshitab/data/SNLI-VE/data/";

{
  "dataset_reader": {
    "type": "visual-entailment",
    "image_dir": datadir + "Flickr30K/flickr30k_images",
    "feature_cache_dir": datadir + "/feature_cache",
    "image_loader": "torch",
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
    "max_instances": 30000,
    "image_processing_batch_size": 16,
  },
  "validation_dataset_reader": self.dataset_reader,
  "train_data_path": datadir + "snli_ve_train.jsonl",
  "validation_data_path": datadir + "snli_ve_dev.jsonl",
  "model": {
    "type": "ve_vilbert_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 1024,
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
    "cuda_devices": std.range(0, num_gpus - 1)
    #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  "trainer": {
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 4e-5
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 2000,
      "num_steps_per_epoch": std.ceil(30000 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"])
    },
    "validation_metric": "+f1",
    "num_epochs": 20,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus)
  },
}

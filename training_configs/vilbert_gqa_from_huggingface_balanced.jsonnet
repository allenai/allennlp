local model_name = "bert-base-uncased";
local effective_batch_size = 128;
local gpu_batch_size = 32;
local num_gpus = 1;

local construct_vocab = false;

#local gqa_dir = "/Users/dirkg/Documents/data/vision/gqa/";
local gqa_dir = "/mnt/tank/dirkg/data/vision/gqa/";

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      "directory": "https://storage.googleapis.com/allennlp-public-data/gqa/vilbert_gqa.vocab.tar.gz"
    };

{
  "dataset_reader": {
    "type": "gqa",
    "image_dir": gqa_dir + "/images",
    [if !construct_vocab then "feature_cache_dir"]: gqa_dir + "/feature_cache",
    [if !construct_vocab then "image_loader"]: "torch",
    [if !construct_vocab then "image_featurizer"]: "resnet_backbone",
    [if !construct_vocab then "region_detector"]: "faster_rcnn",
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
    "image_processing_batch_size": 16,
    "answer_vocab": if construct_vocab then null else vocabulary,
  },
  "validation_dataset_reader": self.dataset_reader {
    "answer_vocab": null
  },
  "vocabulary": vocabulary,
  "train_data_path": "train_balanced",
  "validation_data_path": "testdev_balanced",
  "model": {
    "type": "vqa_vilbert_from_huggingface",
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
    "max_instances_in_memory": 1024*16
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
      "warmup_steps": 5000,
      "num_steps_per_epoch": std.ceil(942255 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"])
    },
    "validation_metric": "+fscore",
    "patience": 5,
    "num_epochs": 20,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus)
  },
}

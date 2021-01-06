local model_name = "bert-base-uncased";
local effective_batch_size = 128;
local gpu_batch_size = 128;
local num_gpus = 1;

local vqa_vocabulary = {
  "type": "from_files",
  "directory": "https://storage.googleapis.com/allennlp-public-data/vqav2/vilbert_vqa_balanced_real.vocab.tar.gz"
};

local reader_common = {
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
    #"max_instances": 1000,
    "image_processing_batch_size": 32,
};

{
  "dataset_reader": {
    "type": "multitask",
    "readers": {
      "vqa": reader_common {
        "type": "vqav2",
        #"image_dir": "/mnt/tank/dirkg/data/vision/vqa/balanced_real",
        #"feature_cache_dir": "/mnt/tank/dirkg/data/vision/vqa/balanced_real/feature_cache",
        "image_dir": "/Users/dirkg/Documents/data/vision/vqa/balanced_real",
        "feature_cache_dir": "/Users/dirkg/Documents/data/vision/vqa/balanced_real/feature_cache",
        "answer_vocab": vqa_vocabulary,
      },
      "gqa": reader_common {
        "type": "gqa",
        #"image_dir": "/mnt/tank/dirkg/data/vision/gqa",
        #"feature_cache_dir": "/mnt/tank/dirkg/data/vision/gqa/feature_cache",
        "image_dir": "/Users/dirkg/Documents/data/vision/gqa",
        "feature_cache_dir": "/Users/dirkg/Documents/data/vision/gqa/feature_cache",
      },
      "ve": reader_common {
        "type": "visual-entailment",
        #"image_dir": "/mnt/tank/dirkg/data/vision/SNLI-VE/data/Flickr30K/flickr30k_images",
        #"feature_cache_dir": "/mnt/tank/dirkg/data/vision/SNLI-VE/data/feature_cache",
        "image_dir": "/Users/dirkg/Documents/data/vision/SNLI-VE/data/Flickr30K/flickr30k_images",
        "feature_cache_dir": "/Users/dirkg/Documents/data/vision/SNLI-VE/data/feature_cache",
      }
    }
  },
  "validation_dataset_reader": self.dataset_reader {
    "readers": super.readers {
      "vqa": super.vqa {
        "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
      }
    }
  },
  "vocabulary": vqa_vocabulary,
  "train_data_path": {
    "vqa": ["balanced_real_train", "balanced_real_val[1000:]"],
    "gqa": "train_balanced",
    "ve": "train",
  },
  "validation_data_path": {
    "vqa": "balanced_real_val[:1000]",
    "gqa": "val_balanced",
    "ve": "dev",
  },
  "model": {
    "type": "multitask",
    "backbone": {
      "type": "vilbert_from_huggingface",
      "model_name": model_name,
      "image_feature_dim": 1024,
      "image_num_hidden_layers": 6,
      "image_hidden_size": 1024,
      "image_num_attention_heads": 8,
      "image_intermediate_size": 1024,
      "image_attention_dropout": 0.1,
      "image_hidden_dropout": 0.1,
      "image_biattention_id": [6, 7, 8, 9, 10, 11],
      "text_biattention_id": [0, 1, 2, 3, 4, 5],
      "text_fixed_layer": 0,
      "image_fixed_layer": 0,
      "combined_hidden_size": 1024,
      "combined_num_attention_heads": 8,
      "pooled_output_dim": 1024,
      "fusion_method": "mul"
    },
    "heads": {
      "vqa": {
        "type": "vqa_vilbert_head"
      },
      "gqa": {
        "type": "vqa_vilbert_head"
      },
      "ve": {
        "type": "ve_vilbert_head"
      }
    }
  },
  "data_loader": {
    "type": "multitask",
    "batch_size": gpu_batch_size,
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-5,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 10000,
    },
    "validation_metric": "+fscore",
    "patience": 5,
    "num_epochs": 30,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus),
  },
}

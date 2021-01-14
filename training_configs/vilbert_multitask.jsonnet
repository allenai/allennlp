local model_name = "bert-base-cased";
local effective_batch_size = 128;
local gpu_batch_size = 128;
local num_gpus = 1;

local construct_vocab = false;

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      "directory": std.format(
        "https://storage.googleapis.com/allennlp-public-data/vilbert/vilbert_multitask.%s.vocab.tar.gz",
        model_name)
    };

local reader_common = {
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
    #"max_instances": 1000, # DEBUG
    "image_processing_batch_size": 32,
};

{
  "dataset_reader": {
    "type": "multitask",
    "readers": {
      "vqa": reader_common {
        "type": "vqav2",
        "image_dir": "/mnt/tank/dirkg/data/vision/vqa/balanced_real",
        [if !construct_vocab then "feature_cache_dir"]: "/mnt/tank/dirkg/data/vision/vqa/balanced_real/feature_cache",
        #"image_dir": "/Users/dirkg/Documents/data/vision/vqa/balanced_real",
        #[if !construct_vocab then "feature_cache_dir"]: "/Users/dirkg/Documents/data/vision/vqa/balanced_real/feature_cache",
        "answer_vocab": if construct_vocab then null else vocabulary,
        "multiple_answers_per_question": !construct_vocab
      },
      "gqa": reader_common {
        "type": "gqa",
        "image_dir": "/mnt/tank/dirkg/data/vision/gqa",
        [if !construct_vocab then "feature_cache_dir"]: "/mnt/tank/dirkg/data/vision/gqa/feature_cache",
        #"image_dir": "/Users/dirkg/Documents/data/vision/gqa",
        #[if !construct_vocab then "feature_cache_dir"]: "/Users/dirkg/Documents/data/vision/gqa/feature_cache",
        "answer_vocab": if construct_vocab then null else vocabulary
      },
      "ve": reader_common {
        "type": "visual-entailment",
        "image_dir": "/mnt/tank/dirkg/data/vision/SNLI-VE/data/Flickr30K/flickr30k_images",
        [if !construct_vocab then "feature_cache_dir"]: "/mnt/tank/dirkg/data/vision/SNLI-VE/data/feature_cache",
        #"image_dir": "/Users/dirkg/Documents/data/vision/SNLI-VE/data/Flickr30K/flickr30k_images",
        #[if !construct_vocab then "feature_cache_dir"]: "/Users/dirkg/Documents/data/vision/SNLI-VE/data/feature_cache",
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
  "vocabulary": vocabulary,
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
    "arg_name_mapping": {
      "backbone": {"question": "text", "hypothesis": "text"}
    },
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
      "image_biattention_id": [0, 1, 2, 3, 4, 5],
      "text_biattention_id": [6, 7, 8, 9, 10, 11],
      "text_fixed_layer": 0,
      "image_fixed_layer": 0,
      "combined_hidden_size": 1024,
      "combined_num_attention_heads": 8,
      "pooled_output_dim": 1024,
      "fusion_method": "mul"
    },
    "heads": {
      "vqa": {
        "type": "vqa",
        "embedding_dim": 1024
      },
      "gqa": {
        "type": "vqa",
        "embedding_dim": 1024
      },
      "ve": {
        "type": "visual_entailment",
        "embedding_dim": 1024
      }
    }
  },
  "data_loader": {
    "type": "multitask",
    "scheduler": {
        "batch_size": gpu_batch_size,
    },
    "shuffle": true,
    //[if !construct_vocab then "max_instances_in_memory"]: 1024*16
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    //"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
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
    "validation_metric": ["+gqa_vqa", "+vqa_vqa", "+ve_acc"],
    "patience": 5,
    "num_epochs": 30,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus),
  },
  "random_seed": 876170670,
  "numpy_seed": 876170670,
  "pytorch_seed": 876170670,
}

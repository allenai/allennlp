local model_name = "bert-base-uncased";
local vocab_size = 30522;     // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local effective_batch_size = 128;
local gpu_batch_size = 128;
local num_gpus = 1;

local construct_vocab = false;
local dataset = "balanced_real";

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      "directory": std.format(
        "https://storage.googleapis.com/allennlp-public-data/vqav2/vilbert_vqa_%s.%s.vocab.tar.gz",
        [dataset, model_name])
    };

{
  "dataset_reader": {
    "type": "vqav2",
    #"feature_cache_dir": "/mnt/tank/dirkg/data/vision/coco/vilbert-multitask-features/fixed-path",
    "feature_cache_dir": "/Users/dirkg/Documents/data/vision/coco/vilbert-multitask-features/fixed-path",
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
    "multiple_answers_per_question": !construct_vocab,
    "write_to_cache": false
  },
  "validation_dataset_reader": self.dataset_reader {
    "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
  },
  "vocabulary": vocabulary,
  "train_data_path": [std.format("%s_train", dataset), std.format("%s_val[1000:]", dataset)],
  "validation_data_path": std.format("%s_val[:1000]", dataset),
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
    "fusion_method": "mul",
    "ignore_text": false, # debug setting
    "ignore_image": false, # debug setting
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    //[if !construct_vocab then "max_instances_in_memory"]: 1024
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-4,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 4e-5}]
      ],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      //"num_steps_per_epoch": std.ceil(0 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"]),
      "warmup_steps": 5000
    },
    "validation_metric": "+fscore",
    "patience": 5,
    "num_epochs": 40,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus),
    "tensorboard_writer": {
        "summary_interval": 10,
        "should_log_learning_rate": true
    },
  },
  "random_seed": 876170670,
  "numpy_seed": 876170670,
  "pytorch_seed": 876170670,
}

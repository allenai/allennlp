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
    "image_dir": std.format("/mnt/tank/dirkg/data/vision/vqa/%s", dataset),
    [if !construct_vocab then "feature_cache_dir"]: std.format("/mnt/tank/dirkg/data/vision/vqa/%s/feature_cache", dataset),
    #"image_dir": std.format("/Users/dirkg/Documents/data/vision/vqa/%s", dataset),
    #[if !construct_vocab then "feature_cache_dir"]: std.format("/Users/dirkg/Documents/data/vision/vqa/%s/feature_cache", dataset),
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
    "image_processing_batch_size": 32,
    "answer_vocab": if construct_vocab then null else vocabulary,
    "multiple_answers_per_question": !construct_vocab
  },
  "validation_dataset_reader": self.dataset_reader {
    "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
  },
  "vocabulary": vocabulary,
  "train_data_path": [std.format("%s_train", dataset), std.format("%s_val[1000:]", dataset)],
  "validation_data_path": std.format("%s_val[:1000]", dataset),
  "model": {
    "type": "vqa_vilbert",
    "text_embeddings": {
      "vocab_size": vocab_size,
      "embedding_size": 768,
      "pad_token_id": 0,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "dropout": 0.1
    },
    "image_embeddings": {
      "feature_size": 1024,
      "embedding_size": 1024
    },
    "encoder": {
      # text
      "hidden_size1": 768,
      "num_hidden_layers1": 12,
      "intermediate_size1": 3072,
      "num_attention_heads1": 12,
      "attention_dropout1": 0.1,
      "hidden_dropout1": 0.1,
      "biattention_id1": [6, 7, 8, 9, 10, 11],
      "fixed_layer1": 0,

      # vision
      "hidden_size2": 1024,
      "num_hidden_layers2": 6,
      "intermediate_size2": 1024,
      "num_attention_heads2": 8,
      "attention_dropout2": 0.1,
      "hidden_dropout2": 0.1,
      "biattention_id2": [0, 1, 2, 3, 4, 5],
      "fixed_layer2": 0,

      "combined_num_attention_heads": 8,
      "combined_hidden_size": 1024,
      "activation": "gelu",
    },
    "pooled_output_dim": 1024,
    "fusion_method": "mul"
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    //[if !construct_vocab then "max_instances_in_memory"]: 1024
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    //"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-4
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      //"num_steps_per_epoch": std.ceil(0 / $["data_loader"]["batch_size"] / $["trainer"]["num_gradient_accumulation_steps"]),
      "warmup_steps": 5000,
    },
    "validation_metric": "+vqa_score",
    "num_epochs": 50,
    "num_gradient_accumulation_steps": effective_batch_size / gpu_batch_size / std.max(1, num_gpus)
  },
  "random_seed": 876170670,
  "numpy_seed": 876170670,
  "pytorch_seed": 876170670,
}

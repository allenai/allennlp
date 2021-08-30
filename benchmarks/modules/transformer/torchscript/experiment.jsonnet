local transformer_model = "bert-base-uncased";

local epochs = 1;
local batch_size = 3;

{
  "dataset_reader": {
      "type": "piqa",
      "transformer_model_name": transformer_model,
  },
  "train_data_path": "torchscript/piqa.jsonl",
  "validation_data_path": "torchscript/piqa.jsonl",
  "model": {
      "type": "transformer_mc",
      "transformer_model": transformer_model
  },
  "data_loader": {
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": 1e-5,
      "eps": 1e-8,
      "correct_bias": true
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 100
    },
    // "grad_norm": 1.0,
    "num_epochs": epochs,
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}

local transformer_model = "bert-base-cased";

{
    "steps": {
        "dataset": {
            "type": "piqa_instances",
            "tokenizer_name": transformer_model
        },
        "loaders": {
            "type": "make_data_loaders",
            "dataset": "dataset",
            "batch_size": 4,
        },
        "trained_model": {
            "type": "training",
            "data_loaders": "loaders",
            "model": {
              "type": "transformer_mc",
              "transformer_model": transformer_model,
              "vocab": "dataset.vocab"
            },
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
            "num_epochs": 20,
            "patience": 3,
            "validation_metric": "+acc",
        }
    }
}

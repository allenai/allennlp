local embedding_dim = 100;
local seq_encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": embedding_dim,
    "num_layers": 1,
    "bidirectional": true
};

{
  "dataset_reader": {
    "type": "snli",
    "token_indexers": {
        "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
        }
    }
  },
  "train_data_path": "allennlp/tests/fixtures/data/esnli_train.jsonl",
  "validation_data_path": "allennlp/tests/fixtures/data/esnli_train.jsonl",
  "model": {
    "type": "esim",
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "allennlp/tests/fixtures/embeddings/glove.6B.100d.sample.txt.gz", //"https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": embedding_dim,
                "trainable": true
            }
        }
    },
    "encoder": seq_encoder,
    "matrix_attention": {
        "type": "dot_product"
    },
    "projection_feedforward": {
        "input_dim": 8*embedding_dim,
        "hidden_dims": embedding_dim,
        "num_layers": 1,
        "activations": "relu"
    },
    "inference_encoder": seq_encoder,
    "output_feedforward": {
        "input_dim": 8*embedding_dim,
        "num_layers": 1,
        "hidden_dims": embedding_dim,
        "activations": "relu",
        "dropout": 0.5
    },
    "output_logit": {
        "input_dim": embedding_dim,
        "num_layers": 1,
        "hidden_dims": 3,
        "activations": "linear"
    },
  },
  "data_loader": {
    "type": "default",
    "batch_size": 10
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 5e-4
    },
  }
}

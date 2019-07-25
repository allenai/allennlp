{
  "dataset_reader": {
    "type": "masked_language_modeling",
  },
  "train_data_path": "allennlp/tests/fixtures/language_model/sentences.txt",
  "validation_data_path": "allennlp/tests/fixtures/language_model/sentences.txt",
  "model": {
    "type": "masked_language_model",
    "target_namespace": "tokens",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 4
        }
      }
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device" : -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}

// The ATIS model uses the preprocessed data from https://github.com/clic-lab/atis
{
  "dataset_reader": {
    "type": "atis",
    "database_file": "/atis/atis.db"
  },
  "validation_dataset_reader": {
      "type": "atis",
      "database_file": "/atis/atis.db",
      "keep_if_unparseable": true
  },
  "train_data_path": "/atis/train.json",
  "validation_data_path": "/atis/dev.json",
  "model": {
    "type": "atis_parser",
    "utterance_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 400,
        "trainable": true
      }
    },
    "action_embedding_dim": 200,
    "encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 800,
      "bidirectional": true,
      "num_layers": 1
    },
    "decoder_beam_search": {
    "beam_size": 10
    },
    "training_beam_size": 1,
    "max_decoding_steps": 200,
    "input_attention": {"type": "dot_product"},
    "dropout": 0.5,
    "database_file": "/atis/atis.db"
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+denotation_acc",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}


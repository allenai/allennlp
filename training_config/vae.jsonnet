local SEED = 0;
local DATA_PATH = "../data/penn/";
local READER = "autoencoder";
local CUDA = -1;

local EMBEDDING_DIM = 512;
local HIDDEN_DIM = 1024;
local LATENT_DIM = 32;
local BATCH_SIZE = 32;

local NUM_EPOCHS = 3;
local SUMMARY_INTERVAL = 1;
local GRAD_CLIPPING = 5;
local GRAD_NORM = 5;
local SHOULD_LOG_PARAMETER_STATISTICS = false;
local SHOULD_LOG_LEARNING_RATE = true;
local OPTIMIZER = "sgd";
local LEARNING_RATE = 1;
local INIT_UNIFORM_RANGE_AROUND_ZERO = 0.1;

local ANNEAL_MIN_WEIGHT = 0.1;
local ANNEAL_MAX_WEIGHT = 1;
local ANNEAL_WARMUP = 0;
local ANNEAL_NUM_ITER_TO_MAX = 5000;
local RECONSTRUCTION_WEIGHT = 1.0;

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": {
    "type": READER
  },
  "train_data_path": DATA_PATH + "/train.txt",
  "validation_data_path": DATA_PATH + "/valid.txt",
  "model": {
    "type": "vae",
    "encoder": {
      "type": "masked_encoder",
      "source_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "tokens",
            "embedding_dim": EMBEDDING_DIM
          }
        }
      },
      "rnn": {
        "type": "lstm",
        "input_size": EMBEDDING_DIM,
        "hidden_size": HIDDEN_DIM,
      }
    },
    "decoder": {
      "type": "variational_decoder",
      "target_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "tokens",
            "embedding_dim": EMBEDDING_DIM
          }
        }
      },
      "rnn": {
        "type": "lstm",
        "input_size": EMBEDDING_DIM + LATENT_DIM,
        "hidden_size": HIDDEN_DIM,
      },
      "latent_dim": LATENT_DIM
    },
    "latent_dim": LATENT_DIM,
    "initializer": [
      [".*", {"type": "uniform", "a": -INIT_UNIFORM_RANGE_AROUND_ZERO, "b": INIT_UNIFORM_RANGE_AROUND_ZERO}]
    ]
  },
  "iterator": {
    "type": "bucket",
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "patience": NUM_EPOCHS,
    "cuda_device": CUDA,
    "summary_interval": SUMMARY_INTERVAL,
    "grad_clipping": GRAD_CLIPPING,
    "grad_norm": GRAD_NORM,
    "loss_weights": {
      "reconstruction": {
        "initial_weight": RECONSTRUCTION_WEIGHT
      },
      "kld": {
        "type": "linear_annealer",
        "min_weight": ANNEAL_MIN_WEIGHT,
        "max_weight": ANNEAL_MAX_WEIGHT,
        "warmup": ANNEAL_WARMUP,
        "num_iter_to_max": ANNEAL_NUM_ITER_TO_MAX,
      },
    },
    "should_log_parameter_statistics": SHOULD_LOG_PARAMETER_STATISTICS,
    "should_log_learning_rate": SHOULD_LOG_LEARNING_RATE,
    "optimizer": {
      "type": OPTIMIZER,
      "lr": LEARNING_RATE
    }
  }
}

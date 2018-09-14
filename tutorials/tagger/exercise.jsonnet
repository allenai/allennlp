// Modification to allow character-level features

local word_embedding_dim = 5;
local char_embedding_dim = 3;
local embedding_dim = word_embedding_dim + char_embedding_dim;
local hidden_dim = 6;
local num_epochs = 1000;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/training.txt',
    "validation_data_path": 'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/validation.txt',
    "dataset_reader": {
        "type": "pos-tutorial",
        "token_indexers": {
            "tokens": { "type": "single_id" },
            "token_characters": { "type": "characters" }
        }
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_embedding_dim
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": char_embedding_dim,
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": char_embedding_dim,
                        "hidden_size": char_embedding_dim
                    }
                }
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["sentence", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}

// jsonnet allows local variables like this
local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 1000;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/training.txt',
    "validation_data_path": 'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/validation.txt',
    "dataset_reader": {
        "type": "pos-tutorial"
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            // Technically you could put a "type": "basic" here,
            // but that's the default TextFieldEmbedder, so doing so
            // is optional.
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
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

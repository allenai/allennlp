// jsonnet allows local variables like this
local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 1000;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'tutorials/tagger/training.txt',
    "validation_data_path": 'tutorials/tagger/validation.txt',
    "dataset_reader": {
        "type": "pos-tutorial"
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
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
        "type": "basic",
        "batch_size": batch_size
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

// Configuration for a basic LSTM sentiment analysis classifier, using the binary Stanford Sentiment Treebank (Socher at al. 2013).
{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "2-class"
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",  
  "model": {
    "type": "basic_classifier",        
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,          
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
          "trainable": false          
          }
      }
    },
    "seq2vec_encoder": {
       "type": "lstm",                     
       "input_size": 300,
       "hidden_size": 512,
       "num_layers": 2,
       "batch_first": true
    }
  },    
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 1,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}

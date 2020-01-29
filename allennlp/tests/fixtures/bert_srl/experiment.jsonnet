local bert_model = "allennlp/tests/fixtures/bert/vocab.txt";
{
  "dataset_reader":{
      "type":"srl",
      "bert_model_name": "bert-base-uncased"
    },
  "train_data_path": "allennlp/tests/fixtures/data/srl",
  "validation_data_path": "allennlp/tests/fixtures/data/srl",
    "model": {
        "type": "srl_bert",
        "bert_model": bert_model,
        "embedding_dropout": 0.0
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 5,
        "padding_noise": 0.0
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}

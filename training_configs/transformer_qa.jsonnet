{
    "steps": {
        "dataset": {
            "type": "hf_dataset",
            "dataset_name": "squad"
        },
        "tokenized": {
            "type": "hf_tokenizer",
            "tokenizer_name": "bert-large-uncased",
            "input": "dataset",
            "fields_to_tokenize": ["answers\\..*", "context", "question"],
            "produce_results": true
        }
    }
}

{
    "steps": {
        "dataset": {
            "type": "huggingface_dataset",
            "dataset_name": "squad"
        },
        "dataset_text_only": {
            "type": "text_only",
            "input": {
                "type": "ref",
                "ref": "dataset"
            },
            "fields_to_keep": ["context", "question"]
        }
    }
}

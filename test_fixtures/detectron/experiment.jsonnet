{
    "dataset_reader": "detectron",
    "train_data_path": "/Users/dirkg/Documents/detectron_datasets/coco_tiny/annotations/instances_minival2014_100.json",
    "validation_data_path": "/Users/dirkg/Documents/detectron_datasets/coco_tiny/annotations/instances_minival2014_100.json",
    "model": {
        "type": "detectron",
    },
    "data_loader": {
       "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5
    }
}

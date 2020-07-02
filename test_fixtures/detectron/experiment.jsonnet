{
    "dataset_reader": {
        "type": "detectron",
        "builtin_config_file": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "overrides": {
            "dataloader": {
                "num_workers": 2
            }
        }
    },
    "train_data_path": "/Users/dirkg/Documents/data/vision/coco_tiny/annotations/instances_minival2014_100.json",
    "validation_data_path": "/Users/dirkg/Documents/data/vision/coco_tiny/annotations/instances_minival2014_100.json",
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

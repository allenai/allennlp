{
    "dataset_reader": {
        "type": "detectron",
        "builtin_config_file": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "image_root": "data/vision/coco/val2014",
        "overrides": {
            "dataloader": {
                "num_workers": 2
            }
        },
    },
    "train_data_path": "data/vision/coco_tiny/annotations/instances_val2014_test.json",
    "validation_data_path": "data/vision/coco_tiny/annotations/instances_val2014_test.json",
    "model": {
        "type": "detectron",
        "train": true,
        "builtin_config_file": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "overrides": {
            "TEST": {
                "DETECTIONS_PER_IMAGE": 36
            },
            "MODEL": {
                "RPN": {
                    "POST_NMS_TOPK_TEST": 300,
                    "NMS_THRESH": 0.7
                },
                "ROI_HEADS": {
                    "NMS_THRESH_TEST" : 0.6,
                    "SCORE_THRESH_TEST" : 0.2,
                },
            },
        },
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

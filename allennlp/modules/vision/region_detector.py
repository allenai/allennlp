from typing import Tuple, List, Any

import torch
import torch.nn.functional as F
from torch import nn, FloatTensor, IntTensor

from allennlp.common.registrable import Registrable


class RegionDetector(nn.Module, Registrable):
    """
    A `RegionDetector` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the format (Batch, #Boxes, 4).
    In other words, for every image, it returns a number of proposed boxes, identified by
    their four coordinates `(x1, y2, x2, y2)`. Coordinates are expected to be between 0
    and 1. Negative coordinates are interpreted as padding.
    """

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> FloatTensor:
        raise NotImplementedError()


@RegionDetector.register("faster_rcnn")
class FasterRcnnRegionDetector(RegionDetector):
    """
    Faster R-CNN (https://arxiv.org/abs/1506.01497) with ResNet backbone.
    Based on detectron2 v0.2
    """

    def __init__(
        self,
        meta_architecture: str = "GeneralizedRCNN",
        device: str = "cpu",
        weights: str = "RCNN-X152-C4-2020-07-18",
        # RPN
        rpn_head_name: str = "StandardRPNHead",
        rpn_in_features: List[str] = ["res4"],
        rpn_boundary_thresh: int = 0,
        rpn_iou_thresholds: List[float] = [0.3, 0.7],
        rpn_iou_labels: List[int] = [0, -1, 1],
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_bbox_reg_loss_type: str = "smooth_l1",
        rpn_bbox_reg_loss_weight: float = 1.0,  # different from default (-1)
        rpn_bbox_reg_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        rpn_smooth_l1_beta: float = 0.1111,  # different from default (0.0)
        rpn_loss_weight: float = 1.0,
        rpn_pre_nms_topk_train: int = 12000,
        rpn_pre_nms_topk_test: int = 6000,
        rpn_post_nms_topk_train: int = 2000,
        rpn_post_nms_topk_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_bbox_loss_weight: float = 1.0,  # not in detectron2 default config

        test_detections_per_image: int = 36,  # different from default (100)
    ):
        super().__init__()

        from allennlp.common import detectron

        flat_parameters = detectron.DetectronFlatParameters(
            meta_architecture=meta_architecture,
            weights=weights,
            device=device,
            rpn_head_name=rpn_head_name,
            rpn_in_features=rpn_in_features,
            rpn_boundary_thresh=rpn_boundary_thresh,
            rpn_iou_thresholds=rpn_iou_thresholds,
            rpn_iou_labels=rpn_iou_labels,
            rpn_batch_size_per_image=rpn_batch_size_per_image,
            rpn_positive_fraction=rpn_positive_fraction,
            rpn_bbox_reg_loss_type=rpn_bbox_reg_loss_type,
            rpn_bbox_reg_loss_weight=rpn_bbox_reg_loss_weight,
            rpn_bbox_reg_weights=rpn_bbox_reg_weights,
            rpn_smooth_l1_beta=rpn_smooth_l1_beta,
            rpn_loss_weight=rpn_loss_weight,
            rpn_pre_nms_topk_train=rpn_pre_nms_topk_train,
            rpn_pre_nms_topk_test=rpn_pre_nms_topk_test,
            rpn_post_nms_topk_train=rpn_post_nms_topk_train,
            rpn_post_nms_topk_test=rpn_post_nms_topk_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_bbox_loss_weight=rpn_bbox_loss_weight,
            test_detections_per_image=test_detections_per_image
        )
        pipeline = detectron.get_pipeline_from_flat_parameters(flat_parameters, make_copy=False)
        self.model = pipeline.model

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> (List[FloatTensor], List[FloatTensor], List[Any]):
        batch_size = len(image_sizes)
        # RPN
        from detectron2.structures import ImageList

        image_list = ImageList(raw_images, [(image[0], image[1]) for image in image_sizes])
        assert len(self.model.proposal_generator.in_features) == 1
        featurized_images_in_dict = {
            self.model.proposal_generator.in_features[0]: featurized_images
        }
        proposals, _ = self.model.proposal_generator(image_list, featurized_images_in_dict, None)
        num_proposals = len(proposals[0])
        assert all([len(p) == num_proposals for p in proposals])
        _, pooled_features = self.model.roi_heads.get_roi_features(
            featurized_images_in_dict, proposals
        )
        predictions = self.model.roi_heads.box_predictor(pooled_features)
        # class probablity
        cls_probs = F.softmax(predictions[0], dim=-1)
        cls_probs = cls_probs[:, :-1]  # background is last

        predictions, r_indices = self.model.roi_heads.box_predictor.inference(
            predictions, proposals
        )

        box_type = type(proposals[0].proposal_boxes)
        batch_boxes = []
        batch_features = []
        batch_probs = []
        feature_dim = pooled_features.size(-1)
        pooled_features = pooled_features.view(batch_size, num_proposals, feature_dim)
        num_classes = cls_probs.size(-1)
        cls_probs = cls_probs.view(batch_size, num_proposals, num_classes)
        for image_num, image_proposal_indices in enumerate(r_indices):
            batch_boxes.append(proposals[image_num].proposal_boxes.tensor[image_proposal_indices])
            batch_features.append(pooled_features[image_num][image_proposal_indices])
            batch_probs.append(cls_probs[image_num][image_proposal_indices])

        features_tensor = torch.stack(batch_features, dim=0)
        boxes_tensor = torch.stack(batch_boxes, dim=0)
        probs_tensor = torch.stack(batch_probs, dim=0)
        return features_tensor, boxes_tensor, probs_tensor

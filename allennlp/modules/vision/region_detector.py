from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, FloatTensor, IntTensor

from allennlp.common.registrable import Registrable


class RegionDetector(nn.Module, Registrable):
    """
    A `RegionDetector` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and finds regions of interest (or "boxes") within those images.

    Those regions of interest are described by three values:
    - A feature vector for each region, which is a tensor of shape (Batch, NumBoxes, FeatureDim)
    - The coordinates of each region within the original image, with shape (Batch, NumBoxes, 4)
    - (Optionally) Class probabilities from some object detector that was used to find the regions
      of interest, with shape (Batch, NumBoxes, NumClasses).

    Because the class probabilities are an optional return value, we return these tensors in a
    dictionary, instead of a tuple (so you don't have a confusing check on the tuple size).  The
    keys are "coordinates", "features", and "class_probs".
    """

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> Dict[str, torch.Tensor]:
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
        detections_per_image: int = 36,  # different from default (100)
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
            test_detections_per_image=detections_per_image,
        )
        pipeline = detectron.get_pipeline_from_flat_parameters(flat_parameters, make_copy=False)
        self.model = pipeline.model
        self.detections_per_image = detections_per_image

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(image_sizes)
        # RPN
        from detectron2.structures import ImageList

        image_list = ImageList(raw_images, [(image[0], image[1]) for image in image_sizes])
        assert len(self.model.proposal_generator.in_features) == 1
        featurized_images_in_dict = {
            self.model.proposal_generator.in_features[0]: featurized_images
        }

        proposals, _ = self.model.proposal_generator(image_list, featurized_images_in_dict, None)

        # this will concatenate the pooled_features from different images. 
        _, pooled_features = self.model.roi_heads.get_roi_features(
            featurized_images_in_dict, proposals
        )
        
        predictions = self.model.roi_heads.box_predictor(pooled_features)
        
        # class probablity
        cls_probs = [F.softmax(prediction, dim=-1) for prediction in predictions]
        cls_probs = [cls_prob[:, :-1] for cls_prob in cls_probs] # background is last

        predictions, r_indices = self.model.roi_heads.box_predictor.inference(
            predictions, proposals
        )
        
        features = []
        proposal_start = 0
        for image_num, image_proposal_indices in enumerate(r_indices):
            feature_cell = {}
            feature_cell['boxes'] = proposals[image_num].proposal_boxes.tensor[image_proposal_indices]
            feature_cell['probs'] = cls_probs[image_num][image_proposal_indices]
            feature_cell['features'] = torch.narrow(pooled_features, 0, proposal_start, len(proposals[image_num]))[image_proposal_indices]
            proposal_start += len(proposals[image_num])
            features.append(feature_cell)
            
        return features

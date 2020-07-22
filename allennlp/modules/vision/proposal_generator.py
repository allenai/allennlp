from typing import Union, List, Callable, Optional, Dict, Any, Tuple

import torch
from torch import nn, FloatTensor

from allennlp.common.registrable import Registrable


class ProposalGenerator(nn.Module, Registrable):
    """
    A `ProposalGenerator` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the format (Batch, #Boxes, 4).
    In other words, for every image, it returns a number of proposed boxes, identified by
    their four coordinates `(x1, y2, x2, y2)`. Coordinates are expected to be between 0
    and 1. Negative coordinates are interpreted as padding.
    """

    def forward(self, images: FloatTensor):
        raise NotImplementedError()


@ProposalGenerator.register("RPN")
class RPNProposalGenerator(ProposalGenerator):
    """A `ProposalGenerator` that never returns any proposals."""
    def __init__(
        self, 
        meta_architecture = "GeneralizedRCNN",
        weights: str = "",
        head_name: str = "StandardRPNHead",
        in_features: List[str] = ["res4"],
        boundary_thresh: int = -1,
        iou_thresholds: List[float] = [0.3, 0.7],
        iou_labels: List[int] = [0, -1, 1],
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        bbox_reg_loss_type: str = 'smooth_l1',
        bbox_reg_loss_weight: float = 1.0,
        bbox_reg_weights: Tuple[float] = (1.0, 1.0, 1.0, 1.0),
        smooth_l1_beta: float = 0.0,
        loss_weight: float = 1.0,
        pre_nms_topk_train: int = 12000,
        pre_nms_topk_test: int = 6000,
        post_nms_topk_train: int = 2000,
        post_nms_topk_test: int = 1000,
        nms_thresh: float = 0.7,
        ):
        super().__init__()
        
        overrides = {
            "MODEL": {
                "WEIGHTS": weights,
                "META_ARCHITECTURE": meta_architecture, 
                "PROPOSAL_GENERATOR": {
                    "NAME": "RPN",
                },
                "RPN":{
                    "HEAD_NAME": head_name, 
                    "IN_FEATURES": in_features, 
                    "BOUNDARY_THRESH": boundary_thresh, 
                    "IOU_THRESHOLDS": iou_thresholds, 
                    "IOU_LABELS": iou_labels, 
                    "BATCH_SIZE_PER_IMAGE": batch_size_per_image, 
                    "POSITIVE_FRACTION": positive_fraction, 
                    "BBOX_REG_LOSS_TYPE":bbox_reg_loss_type, 
                    "BBOX_REG_LOSS_WEIGHT": bbox_reg_loss_weight, 
                    "BBOX_REG_WEIGHTS": bbox_reg_weights, 
                    "SMOOTH_L1_BETA": smooth_l1_beta, 
                    "LOSS_WEIGHT": loss_weight, 
                    "PRE_NMS_TOPK_TRAIN": pre_nms_topk_train, 
                    "PRE_NMS_TOPK_TEST": pre_nms_topk_test, 
                    "POST_NMS_TOPK_TRAIN": post_nms_topk_train, 
                    "POST_NMS_TOPK_TEST": post_nms_topk_test, 
                    "NMS_THRESH": nms_thresh, 
                },
            },
        }

        from allennlp.common.detectron import get_detectron_cfg
        cfg = get_detectron_cfg(None, None, overrides)
        from detectron2.modeling import build_model
        # TODO: Since we use `GeneralizedRCNN` from detectron2, the model initlized 
        # here still has ROI heads and other redunant parameters for RPN.  
        self.model = build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()

    def forward(self, images):
        
        import pdb
        pdb.set_trace()
        features = self.model.backbone(images)
        proposals, _ = self.model.proposal_generator(images, features, None)

        import pdb
        pdb.set_trace()

        return torch.zeros(images.size(0), 0, 4, dtype=torch.float32, device=images.device)

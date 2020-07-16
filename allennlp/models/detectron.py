from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model

from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    build_box_head,
    build_mask_head,
    select_foreground_proposals,
    ROI_HEADS_REGISTRY,
    ROIHeads,
    Res5ROIHeads,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.poolers import ROIPooler

@Model.register("detectron")
class Detectron(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        train: Optional[bool] = True,
        builtin_config_file: Optional[str] = None,
        yaml_config_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        from allennlp.common.detectron import get_detectron_cfg

        cfg = get_detectron_cfg(builtin_config_file, yaml_config_file, overrides=None, freeze=False)
        if train:
            from detectron2.model_zoo import model_zoo

            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(builtin_config_file)
        if overrides is not None:
            from allennlp.common.detectron import apply_dict_to_cfg
            apply_dict_to_cfg(overrides, cfg)

        cfg.freeze()

        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        # register new detectron2 module.
        register_detectron2_module(cfg)

        self.detectron_model = build_model(cfg)
        DetectionCheckpointer(self.detectron_model).load(cfg.MODEL.WEIGHTS)

        from detectron2.utils.events import EventStorage

        self.detectron_event_storage = EventStorage()

    def forward(  # type: ignore
        self, images: Any, proposals: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        self.detectron_event_storage.step()
        self.detectron_model.train()

        with self.detectron_event_storage:
            images = self.detectron_model.preprocess_image(images)
            features = self.detectron_model.backbone(images.tensor)

            if proposals is None:
                proposals, _ = self.detectron_model.proposal_generator(images, features, None)

            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.detectron_model.roi_heads.in_features]

        return {"loss": torch.tensor(0.0)}


class AttributePredictor(nn.Module):
    """
    Head for attribute prediction, including feature/score computation and
    loss computation.
    """
    def __init__(self, cfg, input_dim):
        super().__init__()

        # fmt: off
        self.num_objs          = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.obj_embed_dim     = cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM
        self.fc_dim            = cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM
        self.num_attributes    = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES
        self.max_attr_per_ins  = cfg.INPUT.MAX_ATTR_PER_INS
        self.loss_weight       = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT
        # fmt: on

        # object class embedding, including the background class
        self.obj_embed = nn.Embedding(self.num_objs + 1, self.obj_embed_dim)
        input_dim += self.obj_embed_dim
        self.fc = nn.Sequential(
                nn.Linear(input_dim, self.fc_dim),
                nn.ReLU()
            )
        self.attr_score = nn.Linear(self.fc_dim, self.num_attributes)
        nn.init.normal_(self.attr_score.weight, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, obj_labels):
        attr_feat = torch.cat((x, self.obj_embed(obj_labels)), dim=1)
        return self.attr_score(self.fc(attr_feat))

    def loss(self, score, label):
        n = score.shape[0]
        score = score.unsqueeze(1)
        score = score.expand(n, self.max_attr_per_ins, self.num_attributes).contiguous()
        score = score.view(-1, self.num_attributes)
        inv_weights = (
            (label >= 0).sum(dim=1).repeat(self.max_attr_per_ins, 1).transpose(0, 1).flatten()
        )
        weights = inv_weights.float().reciprocal()
        weights[weights > 1] = 0.
        n_valid = len((label >= 0).sum(dim=1).nonzero())
        label = label.view(-1)
        attr_loss = F.cross_entropy(score, label, reduction="none", ignore_index=-1)
        attr_loss = (attr_loss * weights).view(n, -1).sum(dim=1)

        if n_valid > 0:
            attr_loss = attr_loss.sum() * self.loss_weight / n_valid
        else:
            attr_loss = attr_loss.sum() * 0.
        return {"loss_attr": attr_loss}

class AttributeROIHeads(ROIHeads):
    """
    An extension of ROIHeads to include attribute prediction.
    """
    def forward_attribute_loss(self, proposals, box_features):
        proposals, fg_selection_attributes = select_foreground_proposals(
            proposals, self.num_classes
        )
        attribute_features = box_features[torch.cat(fg_selection_attributes, dim=0)]
        obj_labels = torch.cat([p.gt_classes for p in proposals])
        attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
        attribute_scores = self.attribute_predictor(attribute_features, obj_labels)
        return self.attribute_predictor.loss(attribute_scores, attribute_labels)

@ROI_HEADS_REGISTRY.register()
class AttributeRes5ROIHeads(AttributeROIHeads, Res5ROIHeads):
    """
    An extension of Res5ROIHeads to include attribute prediction. 
    This code was modified based on the repo: 
    https://github.com/vedanuj/grid-feats-vqa/blob/master/grid_feats/roi_heads.py
    """
    def __init__(self, cfg, input_shape):
        super(Res5ROIHeads, self).__init__(cfg)
        
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        assert len(self.in_features) == 1
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, out_channels)

    def forward(self, images, features, proposals, targets=None):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor(feature_pooled)

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, feature_pooled))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_conv5_features(self, features):
        features = [features[f] for f in self.in_features]
        return self.res5(features[0])

    def get_roi_features(self, features, proposals):
        assert len(self.in_features) == 1

        features = [features[f] for f in self.in_features]
        box_features = self._shared_roi_transform(
            features, [x.proposal_boxes for x in proposals]
        )
        pooled_features = box_features.mean(dim=[2, 3])
        return box_features, pooled_features


@ROI_HEADS_REGISTRY.register()
class AttributeStandardROIHeads(AttributeROIHeads, StandardROIHeads):
    """
    An extension of StandardROIHeads to include attribute prediction.
    """
    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.attribute_on        = cfg.MODEL.ATTRIBUTE_ON
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, self.box_head.output_shape.channels)

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals)
            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, box_features))
                del box_features

            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def get_conv5_features(self, features):
        assert len(self.in_features) == 1

        features = [features[f] for f in self.in_features]
        return features[0]



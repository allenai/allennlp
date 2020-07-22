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


@ProposalGenerator.register("butd")
class BottomUpTopDownProposalGenerator(ProposalGenerator):
    def __init__(self)
    """A `ProposalGenerator` that never returns any proposals."""
        from detectron2.data import DatasetMapper

        self.cfg = cfg
        self.mapper = DatasetMapper(cfg)
        from detectron2.modeling import build_model

        self.model = build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()

    def forward(self, images: FloatTensor):
        import pdb
        pdb.set_trace()

        return torch.zeros(images.size(0), 0, 4, dtype=torch.float32, device=images.device)

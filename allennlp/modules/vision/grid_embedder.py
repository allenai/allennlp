from torch import nn, FloatTensor

from allennlp.common.registrable import Registrable


class GridEmbedder(nn.Module, Registrable):
    """
    A `GridEmbedder` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the format
    (Batch, Features, new_height, new_width).

    For every image, it embeds a patch of the image, and returns the embedding
    of the patch. The size of the image might change during this operation.
    """

    def forward(self, images: FloatTensor) -> FloatTensor:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        """
        Returns the output dimension that this `GridEmbedder` uses to represent each
        patch. This is `not` the shape of the returned tensor, but the second dimension
        of that shape.
        """
        raise NotImplementedError

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Returns the dimensionality of the grid that's computed by this `GridEmbedder`.  This is the
        last two dimensions of the output of this modules.
        """
        raise NotImplementedError


@GridEmbedder.register("resnet_backbone")
class ResnetBackbone(Image2ImageModule):
    """Runs an image through resnet, as implemented by Detectron."""

    def __init__(self,
        meta_architecture: str = "GeneralizedRCNN",
        device: str = "cpu",
        weights: str = "RCNN-X152-C4-2020-07-18",

        attribute_on: bool = True,  # not in detectron2 default config
        max_attr_per_ins: int = 16,  # not in detectron2 default config

        stride_in_1x1: bool = False,  # different from default (True)
        num_groups: int = 32,  # different from default (1)
        width_per_group: int = 8,  # different from default (64)
        depth: int = 152,  # different from default (50)
    ):
        super().__init__()
        from allennlp.common import detectron

        flat_parameters = detectron.DetectronFlatParameters(
            max_attr_per_ins=max_attr_per_ins,
            device=device,
            weights=weights,
            meta_architecture=meta_architecture,
            attribute_on=attribute_on,
            stride_in_1x1=stride_in_1x1,
            num_groups=num_groups,
            width_per_group=width_per_group,
            depth=depth)

        pipeline = detectron.get_pipeline_from_flat_parameters(flat_parameters, make_copy=False)
        self.backbone = pipeline.model.backbone

    def forward(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        result = self.backbone(images)
        assert len(result) == 1
        return next(iter(result.values()))

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def get_grid_size(self) -> int:
        raise NotImplementedError

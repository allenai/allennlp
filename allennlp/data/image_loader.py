from os import PathLike
from typing import Union, Sequence, Optional, Tuple

import torch
from torch import FloatTensor, IntTensor

from allennlp.common.registrable import Registrable

OnePath = Union[str, PathLike]
ManyPaths = Sequence[OnePath]

ImagesWithSize = Tuple[FloatTensor, IntTensor]


class ImageLoader(Registrable):
    """
    An `ImageLoader` is a callable that takes as input one or more filenames, and outputs two
    tensors.  The first one contains the images and is of shape (batch, color, height, width).  The
    second one contains the image sizes and is of shape (batch, 2) (where the two dimensions contain
    height and width).
    """

    default_implementation = "detectron"

    def __call__(self, filename_or_filenames: Union[OnePath, ManyPaths]) -> ImagesWithSize:
        if not isinstance(filename_or_filenames, list):
            pixels, sizes = self([filename_or_filenames])  # type: ignore
            return pixels[0], sizes[0]

        from allennlp.common.file_utils import cached_path

        filenames = [cached_path(f) for f in filename_or_filenames]
        return self.load(filenames)

    def load(self, filenames: ManyPaths) -> ImagesWithSize:
        raise NotImplementedError()


# These are in strings to avoid a costly / perhaps uninstalled import of detectron.  This typedef is
# not inplace because it makes the line too long to include flake and mypy ignore statements inline.
# TODO(mattg): Maybe better is to make some vision/ directory that doesn't get imported anywhere by
# default, but allows for putting the detectron imports at the top of the file, because this isn't
# the only place we have this issue.
DetectronInput = Union["DetectronConfig", "DetectronFlatParameters"]  # type: ignore # noqa: F821


@ImageLoader.register("detectron")
class DetectronImageLoader(ImageLoader):
    def __init__(
        self,
        config: Optional[DetectronInput] = None,
    ):
        from allennlp.common.detectron import DetectronConfig, DetectronFlatParameters
        from allennlp.common import detectron

        if config is None:
            pipeline = detectron.get_pipeline_from_flat_parameters(
                make_copy=False, fp=DetectronFlatParameters()
            )
        elif isinstance(config, DetectronConfig):
            pipeline = detectron.get_pipeline(make_copy=False, **config._asdict())
        elif isinstance(config, DetectronFlatParameters):
            pipeline = detectron.get_pipeline_from_flat_parameters(
                make_copy=False, **config._asdict()
            )
        else:
            raise ValueError("Unknown type of `config`")

        self.mapper = pipeline.mapper

    def load(self, filenames: ManyPaths) -> ImagesWithSize:
        images = [{"file_name": str(f)} for f in filenames]
        images = [self.mapper(i) for i in images]

        from detectron2.structures import ImageList

        images = ImageList.from_tensors([image["image"] for image in images])

        return (images.tensor.float() / 256, torch.tensor(images.image_sizes, dtype=torch.int32))  # type: ignore

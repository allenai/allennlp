from os import PathLike
from typing import Union, List, Callable, Optional, Dict, Any, Tuple

import torch
from torch import FloatTensor, IntTensor

from allennlp.common.detectron import DetectronConfig, DetectronFlatParameters
from allennlp.common.registrable import Registrable

OnePath = Union[str, PathLike]
ManyPaths = List[OnePath]

ImagesWithSize = Tuple[FloatTensor, IntTensor]


class ImageLoader(Registrable, Callable[[Union[OnePath, ManyPaths]], FloatTensor]):
    """
    An `ImageLoader` is a callable that takes as input one or more filenames, and outputs an two tensors.
    The first one contains the images and is of shape (batch, color, height, width).
    The second one contains the image sizes and is of shape (batch, [height, width]).
    """

    default_implementation = "detectron"

    def __call__(
        self, filename_or_filenames: Union[OnePath, ManyPaths]
    ) -> Tuple[FloatTensor, IntTensor]:
        if not isinstance(filename_or_filenames, list):
            pixels, sizes = self([filename_or_filenames])
            return pixels[0], sizes[0]

        from allennlp.common.file_utils import cached_path
        filenames = [cached_path(f) for f in filename_or_filenames]
        return self.load(filenames)

    def load(self, filenames: ManyPaths) -> ImagesWithSize:
        raise NotImplementedError()


@ImageLoader.register("detectron")
class DetectronImageLoader(ImageLoader):
    def __init__(
        self,
        config: Optional[Union[DetectronConfig, DetectronFlatParameters]] = None
    ):
        from allennlp.common import detectron
        if config is None:
            pipeline = detectron.get_pipeline_from_flat_parameters(make_copy=False, fp=DetectronFlatParameters())
        elif isinstance(config, DetectronConfig):
            pipeline = detectron.get_pipeline(make_copy=False, **config._asdict())
        elif isinstance(config, DetectronFlatParameters):
            pipeline = detectron.get_pipeline_from_flat_parameters(make_copy=False, **config._asdict())
        else:
            raise ValueError("Unknown type of `config`")

        self.mapper = pipeline.mapper
        self.model = pipeline.model

    def load(self, filenames: ManyPaths) -> ImagesWithSize:
        images = [{"file_name": str(f)} for f in filenames]
        images = [self.mapper(i) for i in images]
        images = self.model.preprocess_image(images)

        return images.tensor, torch.tensor(images.image_sizes, dtype=torch.int32)

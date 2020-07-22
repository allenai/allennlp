from os import PathLike
from typing import Union, List, Callable, Optional, Dict, Any, Tuple

from torch import FloatTensor

from allennlp.common.registrable import Registrable

OnePath = Union[str, PathLike]
ManyPaths = List[OnePath]


class ImageLoader(Registrable, Callable[[Union[OnePath, ManyPaths]], FloatTensor]):
    """
    An `ImageLoader` is a callable that takes as input one or more filenames, and outputs an image tensor of shape
    (batch, color, height, width).
    """

    default_implementation = "detectron"

    def __call__(
        self, filename_or_filenames: Union[OnePath, ManyPaths]
    ) -> Union[FloatTensor, List[FloatTensor]]:
        if not isinstance(filename_or_filenames, list):
            return self([filename_or_filenames])[0]

        from allennlp.common.file_utils import cached_path
        filenames = [cached_path(f) for f in filename_or_filenames]
        return self.load(filenames)

    def load(self, filenames: ManyPaths) -> List[FloatTensor]:
        raise NotImplementedError()


@ImageLoader.register("detectron")
class DetectronImageLoader(ImageLoader):
    def __init__(
        self,
        pixel_mean: List[float] = [103.530, 116.280, 123.675],
        pixel_std: List[float] = [1.0, 1.0, 1.0],
        min_size_train: Tuple[int] = (800,),
        min_size_train_sampling: str = "choice",
        max_size_train: int = 1333,
        min_size_test: int = 800,
        max_size_test: int = 1333,
        crop_enabled: bool = False, 
        crop_type: str = "relative_range",
        crop_size: List[float] = [0.9, 0.9],
        image_format: str = "BGR",
        mask_format: str = "polygon",
    ):

        # constructing the overrides to the cfg file.
        overrides = {
            "MODEL": {
                "PIXEL_MEAN": pixel_mean,
                "PIXEL_STD": pixel_std,
            },
            "INPUT": {
                "MIN_SIZE_TRAIN": min_size_train,
                "MIN_SIZE_TRAIN_SAMPLING": min_size_train_sampling,
                "MAX_SIZE_TRAIN": max_size_train,
                "MIN_SIZE_TEST": min_size_test,
                "MAX_SIZE_TEST": max_size_test,
                "FORMAT": image_format,
                "MASK_FORMAT": mask_format,
                "CROP": {
                    "ENABLED": crop_enabled,
                    "TYPE": crop_type,
                    "SIZE": crop_size,
                },
            },
        }
        from allennlp.common.detectron import get_detectron_cfg
        cfg = get_detectron_cfg(None, None, overrides)
        from detectron2.data import DatasetMapper
        self.mapper = DatasetMapper(cfg)

    def load(self, filenames: ManyPaths) -> List[FloatTensor]:
        images = [{"file_name": str(f)} for f in filenames]
        # TODO: is this efficient enough, the detectron2 impelemntation ? 
        images = [self.mapper(i) for i in images]

        # TODO: for detectronImage loader, do we want to return image tensor or orginial detectron dict format?  
        images = [i["image"] for i in images]
        return images

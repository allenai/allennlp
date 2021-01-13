import glob
import logging
from os import PathLike
from typing import (
    Dict,
    List,
    Union,
    Optional,
    MutableMapping,
    Set,
    Tuple,
    Iterator,
    Iterable,
)
import os

import torch
from torch import Tensor
from tqdm import tqdm
import torch.distributed as dist

from allennlp.common import util
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.lazy import Lazy
from allennlp.common.util import int_to_device
from allennlp.common.file_utils import TensorCache
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.image_loader import ImageLoader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector

logger = logging.getLogger(__name__)


class VisionReader(DatasetReader):
    """
    Base class for dataset readers for vision tasks.

    If you don't specify `image_loader`, `image_featurizer`, and `region_detector`, the reader
    assumes it can get all featurized images from the cache.

    If you don't specify `feature_cache`, the reader will featurize all images using the
    featurization components, and use an internal in-memory cache to catch duplicate
    images.

    If you don't specify either of these things, the reader will not produce featurized images
    at all.

    Parameters
    ----------

    image_dir: `str`
        Path to directory containing image files. The structure of the directory doesn't matter. We
        find images by finding filenames that match `*[image_id].jpg`.
    image_loader : `ImageLoader`, optional
        The image loading component.
    image_featurizer: `Lazy[GridEmbedder]`, optional
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `Lazy[RegionDetector]`, optional
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    tokenizer: `Tokenizer`, optional
        The `Tokenizer` to use to tokenize the text. By default, this uses the tokenizer for
        `"bert-base-uncased"`.
    token_indexers: `Dict[str, TokenIndexer]`, optional
        The `TokenIndexer` to use. By default, this uses the indexer for `"bert-base-uncased"`.
    cuda_device: `Union[int, torch.device]`, optional
        Either a torch device or a GPU number. This is the GPU we'll use to featurize the images.
    max_instances: `int`, optional
        For debugging, you can use this parameter to limit the number of instances the reader
        returns.
    image_processing_batch_size: `int`
        The number of images to process at one time while featurizing. Default is 8.
    write_to_cache: `bool`, optional (default = `True`)
        Allows the reader to write to the cache. Disabling this is useful if you don't want
        to accidentally overwrite a cache you already have, or if you don't have write
        access to the cache you're using.
    """

    def __init__(
        self,
        image_dir: Optional[Union[str, PathLike]],
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        write_to_cache: bool = True,
    ) -> None:
        super().__init__(
            max_instances=max_instances,
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
        )

        # tokenizers and indexers
        if tokenizer is None:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
        self._token_indexers = token_indexers

        if not ((image_loader is None) == (image_featurizer is None) == (region_detector is None)):
            raise ConfigurationError(
                "Please either specify all of image_loader, image_featurizer, and region_detector, "
                "or specify none of them if you don't want to featurize images."
            )

        # feature cache
        self.feature_cache_dir = feature_cache_dir
        self.coordinates_cache_dir = feature_cache_dir
        if feature_cache_dir:
            self.write_to_cache = write_to_cache
        else:
            # If we don't have a cache dir, we use a dict in memory as a cache, so we
            # always write.
            self.write_to_cache = True
        self._feature_cache_instance: Optional[MutableMapping[str, Tensor]] = None
        self._coordinates_cache_instance: Optional[MutableMapping[str, Tensor]] = None

        # image processors
        self.image_loader = None
        if image_loader and image_featurizer and region_detector:
            if cuda_device is None:
                if torch.cuda.device_count() > 0:
                    if util.is_distributed():
                        cuda_device = dist.get_rank() % torch.cuda.device_count()
                    else:
                        cuda_device = 0
                else:
                    cuda_device = -1
            check_for_gpu(cuda_device)
            self.cuda_device = int_to_device(cuda_device)
            logger.info(f"Processing images on device {cuda_device}")

            # image loading and featurizing
            self.image_loader = image_loader
            self.image_loader.device = self.cuda_device
            self._lazy_image_featurizer = image_featurizer
            self._image_featurizer = None
            self._lazy_region_detector = region_detector
            self._region_detector = None
            self.image_processing_batch_size = image_processing_batch_size

        self.produce_featurized_images = False
        if self.feature_cache_dir and self.coordinates_cache_dir:
            logger.info(f"Featurizing images with a cache at {self.feature_cache_dir}")
            self.produce_featurized_images = True
        if image_loader and image_featurizer and region_detector:
            if self.produce_featurized_images:
                logger.info("Falling back to a full image featurization pipeline")
            else:
                logger.info("Featurizing images with a full image featurization pipeline")
                self.produce_featurized_images = True

        if self.produce_featurized_images:
            if image_dir is None:
                if image_loader and image_featurizer and region_detector:
                    raise ConfigurationError("We need an image_dir to featurize images.")
                else:
                    raise ConfigurationError(
                        "We need an image_dir to use a cache of featurized images. Images won't be "
                        "read if they are cached, but we need the image_dir to determine the right "
                        "cache keys from the file names."
                    )

            logger.info("Discovering images ...")
            self.images = {
                os.path.basename(filename): filename
                for extension in {"png", "jpg"}
                for filename in tqdm(
                    glob.iglob(os.path.join(image_dir, "**", f"*.{extension}"), recursive=True),
                    desc=f"Discovering {extension} images",
                )
            }
            logger.info("Done discovering images")

    @property
    def image_featurizer(self) -> Optional[GridEmbedder]:
        if self._image_featurizer is None:
            if self._lazy_image_featurizer is None:
                return None
            self._image_featurizer = self._lazy_image_featurizer.construct().to(self.cuda_device)  # type: ignore
            self._image_featurizer.eval()  # type: ignore[attr-defined]
        return self._image_featurizer  # type: ignore[return-value]

    @property
    def region_detector(self) -> Optional[RegionDetector]:
        if self._region_detector is None:
            if self._lazy_region_detector is None:
                return None
            self._region_detector = self._lazy_region_detector.construct().to(self.cuda_device)  # type: ignore
            self._region_detector.eval()  # type: ignore[attr-defined]
        return self._region_detector  # type: ignore[return-value]

    @property
    def _feature_cache(self) -> MutableMapping[str, Tensor]:
        if self._feature_cache_instance is None:
            if self.feature_cache_dir is None:
                self._feature_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)
                self._feature_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "features"),
                    read_only=not self.write_to_cache,
                )

        return self._feature_cache_instance

    @property
    def _coordinates_cache(self) -> MutableMapping[str, Tensor]:
        if self._coordinates_cache_instance is None:
            if self.coordinates_cache_dir is None:
                self._coordinates_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)  # type: ignore
                self._coordinates_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "coordinates"),  # type: ignore
                    read_only=not self.write_to_cache,
                )

        return self._coordinates_cache_instance

    def _process_image_paths(
        self, image_paths: Iterable[str], *, use_cache: bool = True
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Processes the given image paths and returns featurized images.

        This consumes image paths one at a time, featurizes them either by going to the cache, or
        by running the featurization models, and yields tensors one at a time. It runs the
        featurization pipeline in batches for performance.

        image_paths: `Iterable[str]`
            the image paths to process
        use_cache: `bool`, default = `True`
            Usually the cache behavior is governed by the `write_to_cache` parameter given to
            `__init__()`. But sometimes we want to override this behavior and turn off the
            cache completely. This parameter lets you do that. This is useful for the
            `Predictor`, so we can make predictions without having to touch a cache,
            even if the model was trained with a cache.
        """
        assert self.produce_featurized_images, (
            "For _process_image_paths() to work, we need either a feature cache, or an image loader, "
            "an image featurizer, and a region detector."
        )

        batch: List[Union[str, Tuple[Tensor, Tensor]]] = []
        unprocessed_paths: Set[str] = set()

        def yield_batch():
            # process the images
            paths = list(unprocessed_paths)
            images, sizes = self.image_loader(paths)
            with torch.no_grad():
                images = images.to(self.cuda_device)
                sizes = sizes.to(self.cuda_device)
                featurized_images = self.image_featurizer(images, sizes)
                detector_results = self.region_detector(images, sizes, featurized_images)
                features = detector_results.features
                coordinates = detector_results.boxes

            # store the processed results in memory, so we can complete the batch
            paths_to_tensors = {path: (features[i], coordinates[i]) for i, path in enumerate(paths)}

            # store the processed results in the cache
            if use_cache and self.write_to_cache:
                for path, (features, coordinates) in paths_to_tensors.items():
                    basename = os.path.basename(path)
                    self._feature_cache[basename] = features
                    self._coordinates_cache[basename] = coordinates

            # yield the batch
            for b in batch:
                if isinstance(b, str):
                    yield paths_to_tensors[b]
                else:
                    yield b

        for image_path in image_paths:
            basename = os.path.basename(image_path)
            try:
                if use_cache:
                    features: Tensor = self._feature_cache[basename]
                    coordinates: Tensor = self._coordinates_cache[basename]
                    if len(batch) <= 0:
                        yield features, coordinates
                    else:
                        batch.append((features, coordinates))
                else:
                    # If we're not using the cache, we pretend we had a cache miss here.
                    raise KeyError
            except KeyError:
                if not (self.image_loader and self.region_detector and self.image_featurizer):
                    if use_cache:
                        raise KeyError(
                            f"Could not find {basename} in the feature cache, and "
                            "image featurizers are not defined."
                        )
                    else:
                        raise KeyError(
                            "Reading the feature cache is disabled, and image featurizers "
                            "are not defined. I can't process anything."
                        )
                batch.append(image_path)
                unprocessed_paths.add(image_path)
                if len(unprocessed_paths) >= self.image_processing_batch_size:
                    yield from yield_batch()
                    batch = []
                    unprocessed_paths = set()

        if len(batch) > 0:
            yield from yield_batch()

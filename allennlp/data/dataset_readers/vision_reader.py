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
from allennlp.common.checks import check_for_gpu
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

    Parameters
    ----------

    image_dir: `str`
        Path to directory containing image files. The structure of the directory doesn't matter. We
        find images by finding filenames that match `*[image_id].jpg`.
    image_featurizer: `GridEmbedder`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `RegionDetector`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    tokenizer: `Tokenizer`, optional
    token_indexers: `Dict[str, TokenIndexer]`, optional
    cuda_device: `Union[int, torch.device]`, optional
    max_instances: `int`, optional
    image_processing_batch_size: `int`, optional (default = `8`)
    skip_image_feature_extraction: `bool`, optional (default = `False`)
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        image_loader: ImageLoader,
        image_featurizer: GridEmbedder,
        region_detector: RegionDetector,
        *,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        skip_image_feature_extraction: bool = False,
    ) -> None:
        super().__init__(
            max_instances=max_instances,
            manual_distributed_sharding=True,
            manual_multi_process_sharding=True,
        )

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

        # tokenizers and indexers
        if tokenizer is None:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
        self._token_indexers = token_indexers

        self.skip_image_feature_extraction = skip_image_feature_extraction
        if not skip_image_feature_extraction:
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
            # image loading
            self.image_loader = image_loader
            self.image_featurizer = image_featurizer.to(self.cuda_device)
            self.region_detector = region_detector.to(self.cuda_device)

            # feature cache
            self.feature_cache_dir = feature_cache_dir
            self.coordinates_cache_dir = feature_cache_dir
            self._features_cache_instance: Optional[MutableMapping[str, Tensor]] = None
            self._coordinates_cache_instance: Optional[MutableMapping[str, Tensor]] = None

            self.image_processing_batch_size = image_processing_batch_size

    @property
    def _features_cache(self) -> MutableMapping[str, Tensor]:
        if self._features_cache_instance is None:
            if self.feature_cache_dir is None:
                self._features_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)
                self._features_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "features")
                )

        return self._features_cache_instance

    @property
    def _coordinates_cache(self) -> MutableMapping[str, Tensor]:
        if self._coordinates_cache_instance is None:
            if self.coordinates_cache_dir is None:
                self._coordinates_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)  # type: ignore
                self._coordinates_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "coordinates")  # type: ignore
                )

        return self._coordinates_cache_instance

    def _process_image_paths(
        self, image_paths: Iterable[str], *, use_cache: bool = True
    ) -> Iterator[Tuple[Tensor, Tensor]]:
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
                features = detector_results["features"]
                coordinates = detector_results["coordinates"]

            # store the processed results in memory, so we can complete the batch
            paths_to_tensors = {path: (features[i], coordinates[i]) for i, path in enumerate(paths)}

            # store the processed results in the cache
            if use_cache:
                for path, (features, coordinates) in paths_to_tensors.items():
                    basename = os.path.basename(path)
                    self._features_cache[basename] = features
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
                    features: Tensor = self._features_cache[basename]
                    coordinates: Tensor = self._coordinates_cache[basename]
                    if len(batch) <= 0:
                        yield features, coordinates
                    else:
                        batch.append((features, coordinates))
                else:
                    # If we're not using the cache, we pretend we had a cache miss here.
                    raise KeyError
            except KeyError:
                batch.append(image_path)
                unprocessed_paths.add(image_path)
                if len(unprocessed_paths) >= self.image_processing_batch_size:
                    yield from yield_batch()
                    batch = []
                    unprocessed_paths = set()

        if len(batch) > 0:
            yield from yield_batch()

import glob
import os
from os import PathLike
from typing import Any, Dict, Union, Optional

from overrides import overrides
import torch

from allennlp.common.file_utils import cached_path, json_lines_from_file, TensorCache
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, MetadataField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector


@DatasetReader.register("nlvr2")
class Nlvr2Reader(DatasetReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_loader: `ImageLoader`
        For whatever image preprocessing you want to do.
    image_featurizer: `GridEmbedder`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `RegionDetector`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    data_dir: `str`
        Path to directory containing text files for each dataset split. These files contain
        the sentences and metadata for each task instance.  If this is `None`, we will grab the
        files from the official NLVR github repository.
    feature_cache_dir: `str`, optional
        Path to a directory that will contain a cache of featurized images.
    tokenizer: `Tokenizer`, optional
    token_indexers: `Dict[str, TokenIndexer]`
    lazy : `bool`, optional
        Whether to load data lazily. Passed to super class.
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        image_loader: ImageLoader,
        image_featurizer: GridEmbedder,
        region_detector: RegionDetector,
        *,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        data_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)

        if cuda_device is None:
            from torch import cuda
            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1
        from allennlp.common.checks import check_for_gpu
        check_for_gpu(cuda_device)
        from allennlp.common.util import int_to_device
        self.cuda_device = int_to_device(cuda_device)

        # Paths to data
        if not data_dir:
            github_url = "https://raw.githubusercontent.com/lil-lab/nlvr/"
            nlvr_commit = "68a11a766624a5b665ec7594982b8ecbedc728c7"
            data_dir = f"{github_url}{nlvr_commit}/nlvr2/data"
        self.splits = {
            "dev": f"{data_dir}/dev.json",
            "test": f"{data_dir}/test1.json",
            "train": f"{data_dir}/train.json",
            "balanced_dev": f"{data_dir}/balanced/balanced_dev.json",
            "balanced_test": f"{data_dir}/balanced/balanced_test1.json",
            "unbalanced_dev": f"{data_dir}/balanced/unbalanced_dev.json",
            "unbalanced_test": f"{data_dir}/balanced/unbalanced_test1.json",
        }
        from tqdm import tqdm
        self.images = {
            os.path.basename(filename): filename
            for filename in tqdm(glob.iglob(os.path.join(image_dir, "**", "*.png"), recursive=True), desc="Discovering images")
        }

        # tokenizers and indexers
        if not tokenizer:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
        self._token_indexers = token_indexers

        # image loading
        self.image_loader = image_loader
        self.image_featurizer = image_featurizer.to(self.cuda_device)
        self.region_detector = region_detector.to(self.cuda_device)

        # feature cache
        if feature_cache_dir is None:
            self._features_cache = {}
            self._coordinates_cache = {}
        else:
            os.makedirs(feature_cache_dir, exist_ok=True)
            self._features_cache = TensorCache(os.path.join(feature_cache_dir, "features"))
            self._coordinates_cache = TensorCache(os.path.join(feature_cache_dir, "coordinates"))

    @overrides
    def _read(self, split_or_filename: str):
        filename = self.splits.get(split_or_filename, split_or_filename)

        json_file_path = cached_path(filename)

        json_blob: Dict[str, Any]
        for json_blob in json_lines_from_file(json_file_path):  # type: ignore
            identifier = json_blob["identifier"]
            sentence = json_blob["sentence"]
            label = bool(json_blob["label"])
            instance = self.text_to_instance(identifier, sentence, label)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self, identifier: str, sentence: str, label: bool,  # type: ignore
    ) -> Instance:
        tokenized_sentence = self._tokenizer.tokenize(sentence)

        sentence_field = TextField(tokenized_sentence, self._token_indexers)

        # Load images
        image_name_base = identifier[: identifier.rindex("-")]
        image_paths = [self.images[f"{image_name_base}-{suffix}"] for suffix in ["img0.png", "img1.png"]]

        to_compute = []
        for path in image_paths:
            name = os.path.basename(path)
            if name not in self._features_cache or name not in self._coordinates_cache:
                to_compute.append(path)
        if len(to_compute) > 0:
            images, sizes = self.image_loader(to_compute)
            with torch.no_grad():
                images = images.to(self.cuda_device)
                sizes = sizes.to(self.cuda_device)
                featurized_images = self.image_featurizer(images, sizes)
                detector_results = self.region_detector(images, sizes, featurized_images)
                features = detector_results["features"]
                coordinates = detector_results["coordinates"]

            for index, path in enumerate(to_compute):
                self._features_cache[os.path.basename(path)] = features[index].cpu()
                self._coordinates_cache[os.path.basename(path)] = coordinates[index].cpu()

        left_features, right_features = [
            self._features_cache[os.path.basename(path)]
            for path in image_paths
        ]
        left_coords, right_coords = [
            self._coordinates_cache[os.path.basename(path)]
            for path in image_paths
        ]

        fields = {
            "sentence": sentence_field,
            "box_features": ListField([ArrayField(left_features), ArrayField(right_features)]),
            "box_coordinates": ListField([ArrayField(left_coords), ArrayField(right_coords)]),
            "identifier": MetadataField(identifier),
        }

        if label is not None:
            fields["label"] = LabelField(int(label), skip_indexing=True)

        return Instance(fields)

import glob
import os
from os import PathLike
from typing import Dict, Union

from overrides import overrides
import torch

from allennlp.common.file_utils import cached_path, json_lines_from_file
from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, ListField, MetadataField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.vision import GridEmbedder, RegionDetector


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
        data_dir: Union[str, PathLike] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)

        # Paths to data
        if not data_dir:
            github_url = "https://raw.githubusercontent.com/lil-lab/nlvr/"
            nlvr_commit = "68a11a766624a5b665ec7594982b8ecbedc728c7"
            data_dir  = f"{github_url}{nlvr_commit}/nlvr2/data"
        self.splits = {
            "dev": f"{data_dir}/dev.json",
            "test": f"{data_dir}/test1.json",
            "train": f"{data_dir}/train.json",
            "balanced_dev": f"{data_dir}/balanced/balanced_dev.json",
            "balanced_test": f"{data_dir}/balanced/balanced_test1.json",
            "unbalanced_dev": f"{data_dir}/balanced/unbalanced_dev.json",
            "unbalanced_test": f"{data_dir}/balanced/unbalanced_test1.json",
        }
        self.images = {
            os.path.basename(filename): filename
            for filename in glob.iglob(os.path.join(image_dir, "**", "*.png"), recursive=True)
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
        self.image_featurizer = image_featurizer
        self.region_detector = region_detector

    @overrides
    def _read(self, split_or_filename: str):
        filename = self.splits.get(split_or_filename, split_or_filename)

        json_file_path = cached_path(filename)

        for json in json_lines_from_file(json_file_path):
            identifier = json["identifier"]
            sentence = json["sentence"]
            label = bool(json["label"])
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
        image_name_base = identifier[:identifier.rindex("-")]
        images_path = [self.images[f"{image_name_base}-img{image_id}.png"] for image_id in [0, 1]]

        # TODO: we need a image_reader method that can cache and return feature given a single image path.
        images, sizes = self.image_loader([images_path[0]])
        with torch.no_grad():
            featurized_images = self.image_featurizer(images, sizes)
            pooled_features, bboxes, cls_probs = self.region_detector(
                images, sizes, featurized_images
            )

        fields = {
            "visual_features": ListField([ArrayField(a) for a in visual_features]),
            "box_coordinates": ListField([ArrayField(a) for a in proposals]),
            "sentence": MetadataField(sentence),
            "identifier": MetadataField(identifier),
            "sentence_field": sentence_field,
        }
        return Instance(fields)

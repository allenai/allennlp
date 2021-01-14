from os import PathLike
from typing import (
    Dict,
    Union,
    Optional,
    Tuple,
    Iterable,
)
import json
import os

from overrides import overrides
import torch
from torch import Tensor

from allennlp.common.file_utils import cached_path
from allennlp.common.lazy import Lazy
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector
from allennlp.data.dataset_readers.vision_reader import VisionReader


@DatasetReader.register("gqa")
class GQAReader(VisionReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_loader : `ImageLoader`
    image_featurizer: `Lazy[GridEmbedder]`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `Lazy[RegionDetector]`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    data_dir: `str`
        Path to directory containing text files for each dataset split. These files contain
        the sentences and metadata for each task instance.
    tokenizer: `Tokenizer`, optional
    token_indexers: `Dict[str, TokenIndexer]`
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        answer_vocab: Optional[Union[str, Vocabulary]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        data_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        write_to_cache: bool = True,
    ) -> None:
        super().__init__(
            image_dir,
            image_loader=image_loader,
            image_featurizer=image_featurizer,
            region_detector=region_detector,
            feature_cache_dir=feature_cache_dir,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            cuda_device=cuda_device,
            max_instances=max_instances,
            image_processing_batch_size=image_processing_batch_size,
            write_to_cache=write_to_cache,
        )
        self.data_dir = data_dir

        # read answer vocab
        if answer_vocab is None:
            self.answer_vocab = None
        else:
            if isinstance(answer_vocab, str):
                answer_vocab = cached_path(answer_vocab, extract_archive=True)
                answer_vocab = Vocabulary.from_files(answer_vocab)
            self.answer_vocab = frozenset(
                answer_vocab.get_token_to_index_vocabulary("answers").keys()
            )

    @overrides
    def _read(self, split_or_filename: str):

        if not self.data_dir:
            self.data_dir = "https://nlp.stanford.edu/data/gqa/questions1.2.zip!"

        splits = {
            "challenge_all": f"{self.data_dir}challenge_all_questions.json",
            "challenge_balanced": f"{self.data_dir}challenge_balanced_questions.json",
            "test_all": f"{self.data_dir}test_all_questions.json",
            "test_balanced": f"{self.data_dir}test_balanced_questions.json",
            "testdev_all": f"{self.data_dir}testdev_all_questions.json",
            "testdev_balanced": f"{self.data_dir}testdev_balanced_questions.json",
            "train_balanced": f"{self.data_dir}train_balanced_questions.json",
            "train_all": f"{self.data_dir}train_all_questions",
            "val_all": f"{self.data_dir}val_all_questions.json",
            "val_balanced": f"{self.data_dir}val_balanced_questions.json",
        }

        filename = splits.get(split_or_filename, split_or_filename)
        filename = cached_path(filename, extract_archive=True)

        # If we're considering a directory of files (such as train_all)
        # loop through each in file in generator
        if os.path.isdir(filename):
            files = [os.path.join(filename, file_path) for file_path in os.listdir(filename)]
        else:
            files = [filename]

        # Ensure order is deterministic.
        files.sort()

        for data_file in files:
            with open(data_file) as f:
                questions_with_annotations = json.load(f)

            question_dicts = list(
                self.shard_iterable(
                    questions_with_annotations[q_id] for q_id in questions_with_annotations
                )
            )

            processed_images: Iterable[Optional[Tuple[Tensor, Tensor]]]
            if self.produce_featurized_images:
                # It would be much easier to just process one image at a time, but it's faster to process
                # them in batches. So this code gathers up instances until it has enough to fill up a batch
                # that needs processing, and then processes them all.
                filenames = [f"{question_dict['imageId']}.jpg" for question_dict in question_dicts]
                try:
                    processed_images = self._process_image_paths(
                        self.images[filename] for filename in filenames
                    )
                except KeyError as e:
                    missing_filename = e.args[0]
                    raise KeyError(
                        missing_filename,
                        f"We could not find an image with the name {missing_filename}. "
                        "Because of the size of the image datasets, we don't download them automatically. "
                        "Please download the images from"
                        "https://nlp.stanford.edu/data/gqa/images.zip, "
                        "extract them into a directory, and set the image_dir parameter to point to that "
                        "directory. This dataset reader does not care about the exact directory structure. It "
                        "finds the images wherever they are.",
                    )
            else:
                processed_images = [None] * len(question_dicts)

            for question_dict, processed_image in zip(question_dicts, processed_images):
                answer = {
                    "answer": question_dict["answer"],
                }
                instance = self.text_to_instance(question_dict["question"], processed_image, answer)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image: Optional[Union[str, Tuple[Tensor, Tensor]]],
        answer: Optional[Dict[str, str]] = None,
        *,
        use_cache: bool = True,
    ) -> Optional[Instance]:
        from allennlp.data import Field

        tokenized_question = self._tokenizer.tokenize(question)
        fields: Dict[str, Field] = {"question": TextField(tokenized_question, None)}

        if answer is not None:
            labels_fields = []
            weights = []
            if not self.answer_vocab or answer["answer"] in self.answer_vocab:
                labels_fields.append(LabelField(answer["answer"], label_namespace="answers"))
                weights.append(1.0)

            if len(labels_fields) <= 0:
                return None

            fields["label_weights"] = ArrayField(torch.tensor(weights))
            fields["labels"] = ListField(labels_fields)

        if image is not None:
            if isinstance(image, str):
                features, coords = next(self._process_image_paths([image], use_cache=use_cache))
            else:
                features, coords = image
            fields["box_features"] = ArrayField(features)
            fields["box_coordinates"] = ArrayField(coords)
            fields["box_mask"] = ArrayField(
                features.new_ones((features.shape[0],), dtype=torch.bool),
                padding_value=False,
                dtype=torch.bool,
            )

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["question"].token_indexers = self._token_indexers  # type: ignore

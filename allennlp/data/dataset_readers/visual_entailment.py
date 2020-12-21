import logging

from typing import (
    Dict,
    List,
    Union,
    Optional,
    Tuple,
)

from overrides import overrides
import torch
from torch import Tensor

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.vision_reader import VisionReader

from allennlp.common.file_utils import json_lines_from_file

logger = logging.getLogger(__name__)


@DatasetReader.register("visual-entailment")
class VisualEntailmentReader(VisionReader):
    """
    The dataset reader for visual entailment.
    """

    @overrides
    def _read(self, file_path: str):
        lines = json_lines_from_file(file_path)
        info_dicts: List[Dict] = list(self.shard_iterable(lines))  # type: ignore

        if self.produce_featurized_images:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            filenames = [info_dict["Flickr30K_ID"] + ".jpg" for info_dict in info_dicts]

            try:
                processed_images = self._process_image_paths(
                    [self.images[filename] for filename in filenames]
                )
            except KeyError as e:
                missing_filename = e.args[0]
                raise KeyError(
                    missing_filename,
                    f"We could not find an image with the name {missing_filename}. "
                    "Because of the size of the image datasets, we don't download them automatically. "
                    "Please download the images from"
                    "https://storage.googleapis.com/allennlp-public-data/snli-ve/flickr30k_images.tar.gz, "
                    "extract them into a directory, and set the image_dir parameter to point to that "
                    "directory. This dataset reader does not care about the exact directory structure. It "
                    "finds the images wherever they are.",
                )
        else:
            processed_images = [None for i in range(len(info_dicts))]  # type: ignore

        for info_dict, processed_image in zip(info_dicts, processed_images):
            hypothesis = info_dict["sentence2"]
            answer = info_dict["gold_label"]

            instance = self.text_to_instance(processed_image, hypothesis, answer)
            yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        image: Union[str, Tuple[Tensor, Tensor]],
        hypothesis: str,
        label: Optional[str] = None,
        *,
        use_cache: bool = True,
    ) -> Instance:

        tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)
        hypothesis_field = TextField(tokenized_hypothesis, None)

        fields: Dict[str, Field] = {"hypothesis": hypothesis_field}

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

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["hypothesis"].token_indexers = self._token_indexers  # type: ignore

import logging

from typing import (
    Dict,
    List,
    Union,
    Optional,
    Tuple,
)

from overrides import overrides
from torch import Tensor

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, TextField
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

        if not self.skip_image_feature_extraction:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            processed_images = self._process_image_paths(
                [self.images[info_dict["Flickr30K_ID"] + ".jpg"] for info_dict in info_dicts]
            )
        else:
            processed_images = [None for i in range(len(info_dicts))]  # type: ignore

        attempted_instances_count = 0
        failed_instances_count = 0
        for info_dict, processed_image in zip(info_dicts, processed_images):
            sentence1 = info_dict["sentence1"]
            sentence2 = info_dict["sentence2"]
            answer = info_dict["gold_label"]

            instance = self.text_to_instance(sentence1, sentence2, processed_image, answer)
            attempted_instances_count += 1
            if instance is None:
                failed_instances_count += 1
            else:
                yield instance

            if attempted_instances_count % 2000 == 0:
                failed_instances_fraction = failed_instances_count / attempted_instances_count
                if failed_instances_fraction > 0.1:
                    logger.warning(
                        f"{failed_instances_fraction*100:.0f}% of instances have no answers."
                    )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence1: str,
        sentence2: str,
        image: Union[str, Tuple[Tensor, Tensor]],
        answer: Optional[str] = None,
        *,
        use_cache: bool = True,
    ) -> Optional[Instance]:

        tokenized_sentence1 = self._tokenizer.tokenize(sentence1)
        tokenized_sentence2 = self._tokenizer.tokenize(sentence2)

        sentence1_field = TextField(tokenized_sentence1, None)
        sentence2_field = TextField(tokenized_sentence2, None)

        from allennlp.data import Field

        fields: Dict[str, Field] = {"sentence1": sentence1_field, "sentence2": sentence2_field}

        if image is not None:
            if isinstance(image, str):
                features, coords = next(self._process_image_paths([image], use_cache=use_cache))
            else:
                features, coords = image

            fields["box_features"] = ArrayField(features)
            fields["box_coordinates"] = ArrayField(coords)

        if answer:
            if answer == "-":
                # No gold label could be decided.
                return None

            fields["label"] = LabelField(answer)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["sentence1"].token_indexers = self._token_indexers  # type: ignore
        instance["sentence2"].token_indexers = self._token_indexers  # type: ignore

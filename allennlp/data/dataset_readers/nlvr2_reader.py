from typing import List, Dict, Any
import base64
import csv
import json
import os
import sys
import time

from overrides import overrides
import numpy as np
import spacy

from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    ArrayField,
    LabelField,
    TextField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


FIELDNAMES = [
    "img_id",
    "img_h",
    "img_w",
    "objects_id",
    "objects_conf",
    "attrs_id",
    "attrs_conf",
    "num_boxes",
    "boxes",
    "features",
]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    csv.field_size_limit(sys.maxsize)
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ["img_h", "img_w", "num_boxes"]:
                item[key] = int(item[key])

            boxes = item["num_boxes"]
            decode_config = [
                ("objects_id", (boxes,), np.int64),
                ("objects_conf", (boxes,), np.float32),
                ("attrs_id", (boxes,), np.int64),
                ("attrs_conf", (boxes,), np.float32),
                ("boxes", (boxes, 4), np.float32),
                ("features", (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


@DatasetReader.register("nlvr2_lxmert")
class Nlvr2LxmertReader(DatasetReader):
    """
    Parameters
    ----------
    text_path_prefix: ``str``
        Path to folder containing text files for each dataset split. These files contain
        the sentences and metadata for each task instance.
    visual_path_prefix: ``str``
        Path to folder containing `tsv` files with the extracted objects and visual
        features
    topk_images: ``int``, optional (default=-1)
        Number of images to load from each split's visual features file. If -1, all
        images are loaded
    mask_prepositions_verbs: ``bool``, optional (default=False)
        Whether to mask prepositions and verbs in each sentence
    drop_prepositions_verbs: ``bool``, optional (default=False)
        Whether to drop (remove without replacement) prepositions and verbs in each sentence
    lazy : ``bool``, optional
        Whether to load data lazily.  Passed to super class.
    """

    def __init__(
        self,
        text_path_prefix: str,
        visual_path_prefix: str,
        topk_images: int = -1,
        mask_prepositions_verbs: bool = False,
        drop_prepositions_verbs: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.text_path_prefix = text_path_prefix
        self.visual_path_prefix = visual_path_prefix
        self._tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._token_indexers: Dict[str, TokenIndexer] = {
            "tokens": PretrainedTransformerIndexer("bert-base-uncased")
        }
        self.topk_images = topk_images
        self.mask_prepositions_verbs = mask_prepositions_verbs
        self.drop_prepositions_verbs = drop_prepositions_verbs
        self.image_data: Dict[str, Dict[str, Any]] = {}
        # Loading Spacy to find prepositions and verbs
        self.spacy = spacy.load("en_core_web_sm")

    def get_all_grouped_instances(self, split: str):
        text_file_path = os.path.join(self.text_path_prefix, split + ".json")
        visual_file_path = os.path.join(self.visual_path_prefix, split + ".tsv")
        visual_features = load_obj_tsv(visual_file_path, self.topk_images)
        for img in visual_features:
            self.image_data[img["img_id"]] = img
        instances = []
        with open(text_file_path) as f:
            examples = json.load(f)
            examples_dict = {}
            for example in examples:
                if example["img0"] not in self.image_data or example["img1"] not in self.image_data:
                    continue
                identifier = example["identifier"].split("-")
                identifier = identifier[0] + "-" + identifier[1] + "-" + identifier[-1]
                if identifier not in examples_dict:
                    examples_dict[identifier] = {
                        "sent": example["sent"],
                        "identifier": identifier,
                        "image_ids": [],
                    }
                examples_dict[identifier]["image_ids"] += [
                    example["img0"],
                    example["img1"],
                ]
            for key in examples_dict:
                instances.append(
                    self.text_to_instance(
                        examples_dict[key]["sent"],
                        examples_dict[key]["identifier"],
                        examples_dict[key]["image_ids"],
                        None,
                        None,
                        only_predictions=True,
                    )
                )
        return instances

    @overrides
    def _read(self, split: str):
        text_file_path = os.path.join(self.text_path_prefix, split + ".json")
        visual_file_path = os.path.join(self.visual_path_prefix, split + ".tsv")
        visual_features = load_obj_tsv(visual_file_path, self.topk_images)
        for img in visual_features:
            self.image_data[img["img_id"]] = img
        with open(text_file_path) as f:
            examples = json.load(f)
            for example in examples:
                if example["img0"] not in self.image_data or example["img1"] not in self.image_data:
                    continue
                yield self.text_to_instance(
                    example["sent"],
                    example["identifier"],
                    [example["img0"], example["img1"]],
                    example["label"],
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        identifier: str,
        image_ids: List[str],
        denotation: str = None,
        only_predictions: bool = False,
    ) -> Instance:
        if self.mask_prepositions_verbs:
            doc = self.spacy(question)
            prep_verb_starts = [
                (token.idx, len(token))
                for token in doc
                if token.dep_ == "prep" or token.pos_ == "VERB"
            ]
            new_question = ""
            prev_end = 0
            for (idx, length) in prep_verb_starts:
                new_question += question[prev_end:idx] + self._tokenizer.tokenizer.mask_token
                prev_end = idx + length
            new_question += question[prev_end:]
            question = new_question
        elif self.drop_prepositions_verbs:
            doc = self.spacy(question)
            prep_verb_starts = [
                (token.idx, len(token))
                for token in doc
                if token.dep_ == "prep" or token.pos_ == "VERB"
            ]
            new_question = ""
            prev_end = 0
            for (idx, length) in prep_verb_starts:
                new_question += question[prev_end:idx]
                prev_end = idx + length
            new_question += question[prev_end:]
            question = new_question
        tokenized_sentence = self._tokenizer.tokenize(question)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)

        original_identifier = identifier
        all_boxes = []
        all_features = []
        for key in image_ids:
            img_info = self.image_data[key]
            boxes = img_info["boxes"].copy()
            features = img_info["features"].copy()
            assert len(boxes) == len(features)

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info["img_h"], img_info["img_w"]
            # Dim=1 indices for `boxes`: 0 and 2 are x_min and x_max, respectively;
            # 1 and 3 are y_min and y_max, respectively
            boxes[..., (0, 2)] /= img_w
            boxes[..., (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1 + 1e-5)
            np.testing.assert_array_less(-boxes, 0 + 1e-5)

            all_boxes.append(boxes)
            all_features.append(features)
        features = np.stack(all_features)
        boxes = np.stack(all_boxes)
        fields = {
            "visual_features": ArrayField(features),
            "box_coordinates": ArrayField(boxes),
            "sentence": MetadataField(question),
            "image_id": MetadataField(image_ids),
            "identifier": MetadataField(original_identifier),
            "sentence_field": sentence_field,
        }

        if denotation is not None:
            fields["denotation"] = LabelField(int(denotation), skip_indexing=True)
        return Instance(fields)

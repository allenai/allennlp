from detectron2.config import CfgNode

from glob import glob
from typing import Dict, List, Optional, Any
import base64
import csv
import json
import os
import pickle
import sys
import time
import unicodedata
import json_lines
import random

from overrides import overrides
import numpy as np
import spacy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
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
import pdb

def apply_dict_to_cfg(d: Dict[str, Any], c: CfgNode) -> None:
    for key, value in d.items():
        key = key.upper()
        if isinstance(value, dict):
            apply_dict_to_cfg(value, c[key])
        else:
            c[key] = value

def load_VCR_dataset(dataset_root, image_root, prefix):
    """
    Load a json in VCR's annotation format and convert to Vision and Language Dataset Dict.
    Args:
        annotation_file (str): full path to the VQA json annotation_file.
        image_root (str): the directory of the image or features.
        prefix (str): the name of the dataset (e.g. "vqa_v2_train").
    """

    # load the unisex_names.csv
    unisex_names = []
    with open(os.path.join(os.path.join(dataset_root, "unisex_names.csv"))) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[1] != "name":
                unisex_names.append(row[1])    
    
    task, split = prefix.split('-')
    annotations_jsonpath = os.path.join(dataset_root, 'annotations', '%s.jsonl' %split)
    
    if task == 'Q_A':
        entries = _load_annotationsQ_A(annotations_jsonpath, image_root, unisex_names)
    elif task == 'QA_R':
        entries = _load_annotationsQA_R(annotations_jsonpath, image_root, unisex_names)
    else:
        raise NotImplementedError("task not supported.")

    return entries

def _converId(img_id):

    img_id = img_id.split("-")
    if "train" in img_id[0]:
        new_id = int(img_id[1])
    elif "val" in img_id[0]:
        new_id = int(img_id[1])
    elif "test" in img_id[0]:
        new_id = int(img_id[1])
    else:
        raise NotImplementedError("not implemented.")

    return new_id

def generate_random_name(det_names, unisex_names):
    random_name = []
    for name in det_names:
        if name == "person":
            word = random.choice(unisex_names)
        else:
            word = name
        random_name.append(word)

    return random_name

def replace_det_with_name(inputs, random_names):

    tokens = []
    for w in inputs:
        if isinstance(w, str):
            tokens.append(w)
        else:
            for idx in w:
                word = random_names[idx]
                tokens.append(word)

    return tokens

def _load_annotationsQ_A(annotations_jsonpath, image_root, unisex_names):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            objects = annotation["objects"]
            random_names = generate_random_name(objects, unisex_names)
            # replace question with random names.

            question = replace_det_with_name(annotation["question"], random_names)
            answer_choices = [replace_det_with_name(answer, random_names) for answer in annotation["answer_choices"]]

            if "test" in annotations_jsonpath:
                ans_label = 0
            else:
                ans_label = annotation["answer_label"]

            img_id = _converId(annotation["img_id"])
            img_fn = os.path.join(image_root, annotation["img_fn"])
            anno_id = int(annotation["annot_id"].split("-")[1])

            question = ' '.join(question)
            answer_choices = [' '.join(answer) for answer in answer_choices]

            entries.append(
                {
                    "question": question,
                    "file_name": img_fn,
                    "answers": answer_choices,
                    "metadata_fn": annotation["metadata_fn"],
                    "target": ans_label,
                    "img_id": img_id,
                    "anno_id": anno_id,
                }
            )

    return entries

def _load_annotationsQA_R(annotations_jsonpath, image_root, random_names):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):

            objects = annotation["objects"]
            random_names = generate_random_name(objects, unisex_names)

            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            if "test" in annotations_jsonpath:
                # for each answer
                for answer in annotation["answer_choices"]:
                    question = annotation["question"] + ["[SEP]"] + answer
                    img_id = _converId(annotation["img_id"])
                    ans_label = 0
                    img_fn = os.path.join(image_root, annotation["img_fn"])
                    anno_id = int(annotation["annot_id"].split("-")[1])
                    question = ' '.join(question)
                    rationale_choices = [' '.join(rationale_choice) for rationale_choice in annotation["rationale_choices"]]

                    entries.append(
                        {
                            "question": ' '.join(question),
                            "file_name": img_fn,
                            "answers": rationale_choices,
                            "metadata_fn": annotation["metadata_fn"],
                            "target": ans_label,
                            "img_id": img_id,
                        }
                    )
            else:
                det_names = ""
                question = (
                    annotation["question"]
                    + ["[SEP]"]
                    + annotation["answer_choices"][annotation["answer_label"]]
                )
                ans_label = annotation["rationale_label"]
                # img_fn = annotation["img_fn"]
                img_id = _converId(annotation["img_id"])
                img_fn = os.path.join(image_root, annotation["img_fn"])
                question = ' '.join(question)
                rationale_choices = [' '.join(rationale_choice) for rationale_choice in annotation["rationale_choices"]]

                anno_id = int(annotation["annot_id"].split("-")[1])
                entries.append(
                    {
                        "question": question,
                        "file_name": img_fn,
                        "answers": rationale_choices,
                        "metadata_fn": annotation["metadata_fn"],
                        "target": ans_label,
                        "img_id": img_id,
                        "anno_id": anno_id,
                    }
                )

    return entries

@DatasetReader.register("vcr")
class VCRReader(DatasetReader):
    """
    Parameters
    ----------
    """
    def __init__(
        self, 
        image_root: str, 
        dataset_root: str,
        image_type: str = "raw",
        tokenizer: str = "bert-base-uncased",
        builtin_config_file: Optional[str] = None,
        yaml_config_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._image_root = image_root
        self._dataset_root = dataset_root
        self._image_type = image_type
        self._tokenizer = PretrainedTransformerTokenizer(tokenizer)
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(tokenizer)}
        self.num_labels = 4

        if image_type == 'raw':
            from detectron2.config import get_cfg
            cfg = get_cfg()
            from detectron2.model_zoo import get_config_file
            if builtin_config_file is not None:
                cfg.merge_from_file(get_config_file(builtin_config_file))
            if yaml_config_file is not None:
                cfg.merge_from_file(yaml_config_file)
            if overrides is not None:
                apply_dict_to_cfg(overrides, cfg)
            cfg.freeze()
            from detectron2.data import DatasetMapper
            self.dataset_mapper = DatasetMapper(cfg)

        elif image_type == 'feature':
            pass

    @overrides
    def _read(self, split: str):
        """
        split can be Q_A-train, QA_R-train, Q_A-val, QA_R-val.
        """
        datasets = load_VCR_dataset(self._dataset_root, self._image_root, split)

        for instance_dict in datasets:
            instance = self.text_to_instance(instance_dict, split)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        instance_dict: Dict,
        split: str,
    ) -> Instance:
        from detectron2.data.detection_utils import SizeMismatchError
        from PIL import UnidentifiedImageError
        try:
            instance_dict = self.dataset_mapper(instance_dict)
        except (UnidentifiedImageError, SizeMismatchError):
            return None

        instance_dict['tokenized_sentence'] = self._tokenizer.tokenize(instance_dict['question'])
        sentence_field = TextField(instance_dict['tokenized_sentence'], self._token_indexers)    

        fields = {
            "image": ArrayField(instance_dict['image']),
            "sentence_field": sentence_field,
            "sentence": MetadataField(instance_dict['question']),
            "anno_id": MetadataField(instance_dict['anno_id']),
        }
        
        fields["target"] = LabelField(int(instance_dict['target']), skip_indexing=True)

        return Instance(fields)
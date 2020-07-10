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

def load_vqa_dataset(dataset_root, image_root, prefix):
    """
    Load a json in VQA's annotation format and convert to Vision and Language Dataset Dict.
    Args:
        question_file (str): full path to the VQA json question_file.
        image_root (str): the directory of the image or features.
    """
    question_path = os.path.join(dataset_root, 'annotations', 'v2_OpenEnded_mscoco_%s_questions.json' %prefix)
    questions = sorted(
        json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
    )

    dataset_dicts = []
    if "test" not in prefix:
        ans_path = os.path.join(dataset_root, 'cache', '%s_target.pkl' %prefix)
        # load answer pickle file
        answers = pickle.load(open(ans_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

        for question, answer in zip(questions, answers):
            record = {}
            assert question['question_id'] == answer['question_id']
            record['question'] = question['question']
            record['question_id'] = question['question_id']
            record['labels'] = answer['labels']
            record['scores'] = answer['scores']
            record['file_name'] = os.path.join(image_root, prefix, 'COCO_%s_%012d.jpg'%(prefix,question['image_id']))
            dataset_dicts.append(record)
    else:
        for question in questions:
            record = {}
            record['question'] = question['question']
            record['question_id'] = question['question_id']
            record['file_name'] = os.path.join(image_root, prefix, 'COCO_%s_%012d.jpg'%(prefix,question['image_id']))
            dataset_dicts.append(record)

    return dataset_dicts

@DatasetReader.register("vqav2")
class VQAv2Reader(DatasetReader):
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
        ans2label_path = os.path.join(dataset_root, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataset_root, "cache", "trainval_label2ans.pkl")
        self.ans2label = pickle.load(open(ans2label_path, "rb"))
        self.label2ans = pickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)

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
        split can be train, val, test, trainval, minival.
        """
        if split == 'train':
            datasets = load_vqa_dataset(self._dataset_root, self._image_root, 'train2014')
        elif split == 'val': 
            datasets = load_vqa_dataset(self._dataset_root, self._image_root, 'val2014')
        elif split == 'test':
            datasets = load_vqa_dataset(self._dataset_root, self._image_root, 'test2015')
        elif split == 'trainval':
            datasets_train = load_vqa_dataset(self._dataset_root, self._image_root, 'train2014')
            datasets_val = load_vqa_dataset(self._dataset_root, self._image_root, 'val2014')
            datasets = datasets_train + datasets_val[:-3000]
        elif split == 'minival':
            datasets_val = load_vqa_dataset(self._dataset_root, self._image_root, 'val2014')
            datasets = datasets_val[-3000:]
        else:
            pass

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

        target = torch.zeros(self.num_labels)
        if "test" not in split:
            if len(instance_dict["labels"]):
                labels = torch.from_numpy(np.array(instance_dict["labels"]))
                scores = torch.from_numpy(np.array(instance_dict["scores"], dtype=np.float32))
                if labels is not None:
                    target.scatter_(0, labels, scores)

        fields = {
            "image": ArrayField(instance_dict['image']),
            "sentence_field": sentence_field,
            "sentence": MetadataField(instance_dict['question']),
            "question_id": MetadataField(instance_dict['question_id']),
            "label": ArrayField(target),
        }

        return Instance(fields)
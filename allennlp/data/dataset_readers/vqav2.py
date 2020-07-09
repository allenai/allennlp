from glob import glob
from typing import Dict, List
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
            dataset_dicts.append(record)
    else:
        for question in questions:
            record = {}
            record['question'] = question['question']
            record['question_id'] = question['question_id']
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
        annotation_root: str,
        tokenizer: str = "bert-base-uncased",
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.image_root = image_root
        self.annotation_root = annotation_root
        self._tokenizer = PretrainedTransformerTokenizer(tokenizer)
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(tokenizer)}

    @overrides
    def _read(self, split: str):
        """
        split can be train, val, test, trainval, minival.
        """
        if split == 'train':
            datasets = load_vqa_dataset(self.annotation_root, self.image_root, 'train2014')
        elif split == 'val': 
            datasets = load_vqa_dataset(self.annotation_root, self.image_root, 'val2014')
        elif split == 'test':
            datasets = load_vqa_dataset(self.annotation_root, self.image_root, 'test2015')
        elif split == 'trainval':
            datasets_train = load_vqa_dataset(self.annotation_root, self.image_root, 'train2014')
            datasets_val = load_vqa_dataset(self.annotation_root, self.image_root, 'val2014')
            datasets = datasets_train + datasets_val[:-3000]
        elif split == 'minival':
            datasets_val = load_vqa_dataset(self.annotation_root, self.image_root, 'val2014')
            datasets = datasets_val[-3000:]
        else:
            pass

        for example in datasets:
            pdb.set_trace()
        
    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image_ids: List[str],
    ) -> Instance:
        pdb.set_trace()
        pass

if __name__ == "__main__":
    # run the tool script to get answer meta_file.
    pass

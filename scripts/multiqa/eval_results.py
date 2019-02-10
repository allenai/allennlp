import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random, math
import sys, os
import boto3
from typing import TypeVar,Iterable
from multiprocessing import Pool
from allennlp.common.elastic_logger import ElasticLogger

T = TypeVar('T')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.common.util import add_noise_to_dict_values

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
import string

def parse_filename(filename):
    results_dict = {}
    match_results = re.match('(\S+)_dev_on_(\S+)_from_(\S+)_(\S+).json', filename)
    if match_results is not None:
        results_dict['eval_set'] = match_results[1]
        results_dict['target_dataset'] = match_results[2]
        results_dict['source_dataset'] = match_results[3]
        results_dict['type'] = match_results[4]
        return results_dict

    logger.error('could not find any parsing for the format %s',filename)
    return

def process_results(filename, type, source_dataset, \
                        target_dataset, eval_set, model, target_size):
    with open(filename, 'r') as f:
        results_dict = json.load(f)
        results_dict['type'] = type
        if 'source_dataset' is not None:
            results_dict['source_dataset'] = source_dataset
        results_dict['target_dataset'] = target_dataset
        results_dict['eval_set'] = eval_set
        results_dict['model'] = model
        if 'target_size' is not None:
            results_dict['target_size'] = target_size
        ElasticLogger().write_log('INFO', 'EvalResults', context_dict=results_dict)

def main():
    parse = argparse.ArgumentParser("Pre-process for DocumentQA/MultiQA model and datareader")
    parse.add_argument("--eval_res_file",default=None, type=str)
    parse.add_argument("--type", default=None, type=str)
    parse.add_argument("--source_dataset", default=None, type=str)
    parse.add_argument("--target_dataset", default=None, type=str)
    parse.add_argument("--eval_set", default=None, type=str)
    parse.add_argument("--model", default=None, type=str)
    parse.add_argument("--target_size", default=None, type=str)

    args = parse.parse_args()

    if args.eval_res_file is not None:
        process_results(args.eval_res_file, args.type, args.source_dataset, \
                        args.target_dataset, args.eval_set, args.model ,args.target_size)
    else:
        logger.error('No input provided')


if __name__ == "__main__":
    main()



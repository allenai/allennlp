import argparse
import json
import logging
import zipfile, gzip, re, copy, random, math
import sys, os
import numpy
from typing import TypeVar,Iterable
from allennlp.common.elastic_logger import ElasticLogger
from subprocess import Popen,call

T = TypeVar('T')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import numpy as np


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
                        target_dataset, eval_set, split_type, model, target_size, \
                    experiment, full_experiments_name, eval_path):
    # computing
    with open(filename, 'r') as f:
        results_dict = json.load(f)

    results_dict['type'] = type
    if 'source_dataset' is not None:
        results_dict['source_dataset'] = source_dataset
    results_dict['target_dataset'] = target_dataset
    results_dict['eval_set'] = eval_set
    results_dict['split_type'] = split_type
    results_dict['model'] = model
    results_dict['experiment'] = experiment
    results_dict['full_experiments_name'] = full_experiments_name
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
    parse.add_argument("--split_type", default='dev', type=str)
    parse.add_argument("--model", default=None, type=str)
    parse.add_argument("--target_size", default=None, type=str)
    parse.add_argument("--experiment", default=None, type=str)
    parse.add_argument("--full_experiments_name", default=None, type=str)
    parse.add_argument("--eval_path", default=None, type=str)
    args = parse.parse_args()


    if args.eval_res_file is not None:
        process_results(args.eval_res_file, args.type, args.source_dataset, \
                        args.target_dataset, args.eval_set, args.split_type, args.model ,args.target_size, \
                        args.experiment, args.full_experiments_name, args.eval_path)
    else:
        logger.error('No input provided')


if __name__ == "__main__":
    main()



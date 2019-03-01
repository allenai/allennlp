import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random, math
import sys, os
import numpy
from typing import TypeVar,Iterable
from multiprocessing import Pool
from allennlp.common.elastic_logger import ElasticLogger
from subprocess import Popen,call

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
                        target_dataset, eval_set, split_type, model, target_size, experiment, full_experiments_name, predictions_file):
    # for BERTlarge we process a precdiction file ...
    if predictions_file is not None:
        instance_list = []
        with open(predictions_file, 'r') as f:
            for line in f:
                try:
                    instance_list.append(json.loads(line))
                except:
                    pass

        instance_list = sorted(instance_list, key=lambda x: x['question_id'])
        intances_question_id = [instance['question_id'] for instance in instance_list]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in range(len(split_inds) - 1)]
        print(len(per_question_instances))
        results_dict = {'EM':0.0, 'f1': 0.0}
        for question_instances in per_question_instances:
            best_ind = numpy.argmax([instance['best_span_logit'] for instance in question_instances])
            results_dict['EM'] += question_instances[best_ind]['EM']
            results_dict['f1'] += question_instances[best_ind]['f1']
        results_dict['EM'] /= len(per_question_instances)
        results_dict['f1'] /= len(per_question_instances)
        results_dict['EM'] *= instance_list[0]['qas_used_fraction']
        results_dict['f1'] *= instance_list[0]['qas_used_fraction']

        # sanity test:
        single_file_path = cached_path('s3://multiqa/datasets/' + eval_set  + '_dev.jsonl.zip')
        all_question_ids = []
        with zipfile.ZipFile(single_file_path, 'r') as myzip:
            if myzip.namelist()[0].find('jsonl') > 0:
                contexts = []
                with myzip.open(myzip.namelist()[0]) as myfile:
                    header = json.loads(myfile.readline())['header']
                    for example in myfile:
                        context = json.loads(example)
                        contexts.append(context)
                        all_question_ids += [qa['id'] for qa in context['qas']]
        predictions_question_ids = list(set(intances_question_id))
        print(set(all_question_ids) - set(predictions_question_ids))
        results_dict['qids_missing_frac'] = len(set(all_question_ids) - set(predictions_question_ids)) / len(set(all_question_ids))




    else:
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

    if predictions_file is not None:
        # uploading to cloud
        command = "aws s3 cp " + predictions_file + " s3://multiqa/predictions/" + predictions_file.split('/')[-1] + " --acl public-read"
        Popen(command, shell=True, preexec_fn=os.setsid)

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
    parse.add_argument("--predictions_file", default=None, type=str)
    args = parse.parse_args()


    if args.eval_res_file is not None:
        process_results(args.eval_res_file, args.type, args.source_dataset, \
                        args.target_dataset, args.eval_set, args.split_type, args.model ,args.target_size, \
                        args.experiment, args.full_experiments_name, args.predictions_file)
    else:
        logger.error('No input provided')


if __name__ == "__main__":
    main()



import argparse
import logging
import os
import shutil
import sys
from subprocess import run

import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.models.archival import load_archive

DEFAULT_EXECUTOR_JAR = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-executor-0.1.0.jar"
ABBREVIATIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"


def main(args: argparse.Namespace):
    predict_logical_forms(args)
    evaluate_logical_forms(args)


def predict_logical_forms(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    archive = load_archive(args.model_archive, cuda_device=args.cuda_device)
    model = archive.model
    model.eval()
    # TODO(mattg): TOTAL HACK! Not sure why I'm getting this - might be due to numbers?
    model.vocab._token_to_index['rule_labels']['@@UNKNOWN@@'] = 0
    config = archive.config
    config['dataset_reader']['lazy'] = True
    dataset_reader = DatasetReader.from_params(config['dataset_reader'])
    dataset = dataset_reader.read(args.test_data)
    data_iterator = BasicIterator(args.batch_size)
    data_iterator.index_with(model.vocab)
    num_batches = data_iterator.get_num_batches(dataset)
    logical_forms = []
    for batch in tqdm.tqdm(data_iterator(dataset, num_epochs=1, shuffle=False)):
        if 'target_action_sequences' in batch:
            # This makes the model skip the loss computation, which will make things a bit faster.
            del batch['target_action_sequences']
        results = model(**batch)
        logical_forms.extend(results['logical_form'])
    logical_form_filename = os.path.join(args.output_dir, 'logical_forms.txt')
    with open(logical_form_filename, 'w') as logical_form_file:
        for logical_form in logical_forms:
            logical_form_file.write(f"{logical_form}\n")


def evaluate_logical_forms(args: argparse.Namespace):
    logical_form_filename = os.path.join(args.output_dir, 'logical_forms.txt')
    # Sempre relies on having a few files available in data/ in the current directory.  It's easier
    # to just download them here instead of trying to change the sempre code...
    keep_data_dir = False
    if os.path.exists('data'):
        if not os.path.isdir('data'):
            raise RuntimeError("A file named 'data' exists in the current directory; can't proceed")
        else:
            keep_data_dir = True
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/abbreviations.tsv'):
        run(f'wget {ABBREVIATIONS_FILE}', shell=True)
        run('mv wikitables-abbreviations.tsv data/abbreviations.tsv', shell=True)
    if not os.path.exists('data/grow.grammar'):
        run(f'wget {GROW_FILE}', shell=True)
        run('mv wikitables-grow.grammar data/grow.grammar', shell=True)
    command = ' '.join(['java',
                        '-jar',
                        cached_path(args.executor_jar),
                        args.test_data,
                        logical_form_filename,
                        args.table_directory,
                        ])
    run(command, shell=True)
    if not keep_data_dir:
        shutil.rmtree('data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate WikiTableQuestions model")
    parser.add_argument('model_archive', type=str, help='The archived model to evaluate')
    parser.add_argument('test_data', type=str, help='The (lisp-formatted) dataset to evaluate on')
    parser.add_argument('table_directory', type=str, help='Base directory for reading tables')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Location to store prediction output')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size when making predictions')
    parser.add_argument('--executor-jar', type=str, default=DEFAULT_EXECUTOR_JAR,
                        help='path to jar for executing logical forms with SEMPRE')
    args = parser.parse_args()
    main(args)

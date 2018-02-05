"""
The ``elmo`` subcommand allows you to make bulk ELMo predictions.

Given a pre-processed input text file, this command outputs the internal
layers used to compute ELMo representations to a single (potentially large) file.

The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (http://docs.h5py.org/en/latest/) where each
sentence is a size (3, num_tokens, 1024) array with the biLM representations.

In the default setting, each sentence is keyed in the output file by the line number
in the original text file.  Optionally, by specifying --use_sentence_key
the first token in each sentence is assumed to be a unique sentence key
used in the output file.
"""

import argparse
from typing import IO

import h5py
import logging
import torch

from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm
from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class Elmo(Subcommand):

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = '''Create word vectors using ELMo.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')
        subparser.add_argument('output_file', type=str, help='path to output file')

        subparser.add_argument('--vocab_path', type=str, help='A path to a vocabulary file to generate ')
        subparser.add_argument(
            '--options_file',
            type=str,
            default=DEFAULT_OPTIONS_FILE,
            help='The path to the ELMo options file.')
        subparser.add_argument(
            '--weight_file',
            type=str,
            default=DEFAULT_WEIGHT_FILE,
            help='The path to the ELMo weight file.')
        subparser.add_argument('--batch_size', type=int, default=64, help='The batch size to use.')
        subparser.add_argument('--cuda_device', type=int, default=-1, help='The cuda_device to run on.')
        subparser.add_argument('--use_sentence_key', default=False, action='store_true')

        subparser.set_defaults(func=elmo_command)

        return subparser


def batch_to_ids(indexer, batch):
    """
    Given a batch (as list of tokenized sentences), return a batch
    of padded character ids.
    """
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']


def batch_to_embeddings(indexer, batch, elmo_bilm, cuda_device):
    # returns (batch_size, 3, num_times, 1024) embeddings and (batch_size, num_times) mask
    character_ids = batch_to_ids(indexer, batch)
    if cuda_device >= 0:
        character_ids = character_ids.cuda(cuda_device=cuda_device)

    bilm_output = elmo_bilm(character_ids)
    layer_activations = bilm_output['activations']
    mask_with_bos_eos = bilm_output['mask']

    without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
            for layer in layer_activations]
    # without_bos_eos is a 3 element list of (batch_size, num_times, dim) arrays
    activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
    mask = without_bos_eos[0][1]

    return activations, mask


def write_batch(indexer, batch, keys, elmo_bilm, cuda_device, fout):
    embeddings, mask = batch_to_embeddings(indexer, batch, elmo_bilm, cuda_device)
    for i, key in enumerate(keys):
        length = int(mask[i, :].sum())
        sentence_embeds = embeddings[i, :, :length, :].data.cpu().numpy()

        ds = fout.create_dataset(key,
                sentence_embeds.shape, dtype='float32',
                data=sentence_embeds
        )


def elmo_command(args):
    elmo(args.options_file,
    args.weight_file,
    args.input_file,
    args.output_file,
    args.batch_size,
    args.cuda_device,
    args.use_sentence_key)


def elmo(options_file: str,
         weight_file: str,
         input_file: IO,
         output_file_path: str,
         batch_size: int,
         cuda_device: int,
         use_sentence_key: bool = False):

    logger.info("Initializing ELMo.")
    elmo_bilm = _ElmoBiLm(options_file, weight_file)
    indexer = ELMoTokenCharactersIndexer()
    if cuda_device >= 0:
        elmo_bilm.cuda(cuda_device=cuda_device)

    logger.info("Processing sentences.")
    with h5py.File(output_file_path, 'w') as fout:
        batch = []
        keys = []
        line_no = 0

        for line in Tqdm.tqdm(input_file):
            tokens = line.strip().split()

            if use_sentence_key:
                keys.append(tokens[0])
                batch.append(tokens[1:])
            else:
                keys.append(str(line_no))
                batch.append(tokens)
            line_no += 1

            if len(batch) >= batch_size:
                # run biLM and save to file
                write_batch(indexer, batch, keys, elmo_bilm, cuda_device, fout)
                batch = []
                keys = []

        if len(batch) > 0:
            write_batch(indexer, batch, keys, elmo_bilm, cuda_device, fout)

    input_file.close()
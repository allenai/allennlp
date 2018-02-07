"""
The ``elmo`` subcommand allows you to make bulk ELMo predictions.

Given a pre-processed input text file, this command outputs the internal
layers used to compute ELMo representations to a single (potentially large) file.

The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (<http://docs.h5py.org/en/latest/>) where each
sentence is a size (3, num_tokens, 1024) array with the biLM representations.

In the default setting, each sentence is keyed in the output file by the line number
in the original text file.  Optionally, by specifying --use-sentence-key
the first token in each sentence is assumed to be a unique sentence key
used in the output file.

#TODO(michaels) add a link to the ELMo paper once published.

.. code-block:: bash

   $ python -m allennlp.run elmo --help
   usage: python -m allennlp.run [command] elmo [-h] [--vocab-path VOCAB_PATH]
                                                [--options-file OPTIONS_FILE]
                                                [--weight-file WEIGHT_FILE]
                                                [--batch-size BATCH_SIZE]
                                                [--cuda-device CUDA_DEVICE]
                                                [--use-sentence-key]
                                                input_file output_file

   Create word vectors using ELMo.

   positional arguments:
     input_file            path to input file
     output_file           path to output file

   optional arguments:
     -h, --help            show this help message and exit
     --vocab-path VOCAB_PATH
                           A path to a vocabulary file to generate
     --options-file OPTIONS_FILE
                           The path to the ELMo options file.
     --weight-file WEIGHT_FILE
                           The path to the ELMo weight file.
     --batch-size BATCH_SIZE
                           The batch size to use.
     --cuda-device CUDA_DEVICE
                           The cuda_device to run on.
     --use-sentence-key
"""

import logging
from typing import IO, List, Iterable, Tuple

import argparse
import h5py
import torch

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm
from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64


class Elmo(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Create word vectors using ELMo.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')
        subparser.add_argument('output_file', type=str, help='path to output file')

        subparser.add_argument('--vocab-path', type=str, help='A path to a vocabulary file to generate ')
        subparser.add_argument(
                '--options-file',
                type=str,
                default=DEFAULT_OPTIONS_FILE,
                help='The path to the ELMo options file.')
        subparser.add_argument(
                '--weight-file',
                type=str,
                default=DEFAULT_WEIGHT_FILE,
                help='The path to the ELMo weight file.')
        subparser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='The batch size to use.')
        subparser.add_argument('--cuda-device', type=int, default=-1, help='The cuda_device to run on.')
        subparser.add_argument('--use-sentence-key', default=False, action='store_true')

        subparser.set_defaults(func=elmo_command)

        return subparser


class ElmoEmbedder():
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        logger.info("Initializing ELMo.")
        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(cuda_device=cuda_device)

        self.cuda_device = cuda_device

    def batch_to_ids(self, batch: List[List[str]]) -> torch.Tensor:
        """
        Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
        (len(batch), max sentence length, max word length).

        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A tensor of padded character ids.
        """
        instances = []
        for sentence in batch:
            tokens = [Token(token) for token in sentence]
            field = TextField(tokens,
                              {'character_ids': self.indexer})
            instance = Instance({"elmo": field})
            instances.append(instance)

        dataset = Batch(instances)
        vocab = Vocabulary()
        dataset.index_instances(vocab)
        return dataset.as_tensor_dict()['elmo']['character_ids']

    def batch_to_embeddings(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids = self.batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(cuda_device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of pairs of (batch_size, num_timesteps, dim) tensors.
        without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
        # The mask is the same for each ELMo vector, so just take the first.
        mask = without_bos_eos[0][1]

        return activations, mask

    def embed_sentence(self, sentence: List[str]) -> torch.Tensor:
        """
        Computes the ELMo embeddings for a single tokenized sentence.

        Parameters
        ----------
        sentence : ``List[str]``, required
            A tokenized sentence.

        Returns
        -------
        A tensor containing the ELMo vectors.
        """

        return self.embed_batch([sentence])[0]

    def embed_batch(self, batch: List[List[str]]) -> List[torch.Tensor]:
        """
        Computes the ELMo embeddings for a batch of tokenized sentences.

        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        elmo_embeddings = []

        embeddings, mask = self.batch_to_embeddings(batch)
        for i in range(len(batch)):
            length = int(mask[i, :].sum())
            sentence_embeds = embeddings[i, :, :length, :].data.cpu().numpy()
            elmo_embeddings.append(sentence_embeds)

        return elmo_embeddings

    def embed_sentences(self, sentences: Iterable[List[str]], batch_size: int) -> Iterable[torch.Tensor]:
        """
        Computes the ELMo embeddings for a iterable of sentences.

        Parameters
        ----------
        sentences : ``Iterable[List[str]]``, required
            An iterable of tokenized sentences.
        batch_size : ``int``, required
            The number of sentences ELMo should process at once.

        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch(batch)

    def embed_file(self,
                   input_file: IO,
                   output_file_path: str,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   use_sentence_key: bool = False) -> None:
        """
        Computes ELMo embeddings from an input_file where each line contains a sentence tokenized by whitespace.
        The ELMo embeddings are written out in HDF5 format, where each sentences is saved in a dataset corresponds
        with a organized by key.  Unless use_sentence_key is set, the key will be the index of the sentence.

        Parameters
        ----------
        input_file : ``IO``, required
            A file with one tokenized sentence per line.
        output_file_path : ``str``, required
            A path to the output hdf5 file.
        batch_size : ``int``, optional, (default = 64)
            The number of sentences to process in ELMo at one time.
        use_sentence_key : ``bool``, optional, (default = False)
            If true, use the first token in each line as the unique key for the layers output to the HDF5 file.
            This key will be stripped from the rest of the line and the remaining tokens will be used as the
            sentence to compute embeddings from.
        """

        # Tokenizes the sentences.
        sentences = [line.strip().split() for line in input_file]
        if use_sentence_key:
            # Uses the first token in each sentence as the key.
            keys, sentences = zip(*[(tokens[0], tokens[1:]) for tokens in sentences])
            embedded_sentences = zip(keys, self.embed_sentences(sentences, batch_size))
        else:
            # Uses the index as the key.
            embedded_sentences = enumerate(self.embed_sentences(sentences, batch_size))

        logger.info("Processing sentences.")
        with h5py.File(output_file_path, 'w') as fout:
            for key, embeddings in Tqdm.tqdm(embedded_sentences):
                fout.create_dataset(
                        str(key),
                        embeddings.shape, dtype='float32',
                        data=embeddings
                )
        input_file.close()

def elmo_command(args):
    elmo_embedder = ElmoEmbedder(args.options_file, args.weight_file, args.cuda_device)
    elmo_embedder.embed_file(
            args.input_file,
            args.output_file,
            args.batch_size,
            args.use_sentence_key)

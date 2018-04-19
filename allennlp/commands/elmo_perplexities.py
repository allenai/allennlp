"""
Computes ELMo perplexities for a bunch of sentences.

Given a pre-processed input text file, this command outputs the perplexities. The output will be a hdf5 file
containing a (2, num_tokens) array with the forward and backward perplexities

WARNING #1: This is *NOT* the same thing as an Elmo embedder.
If that's what you're after, see elmo.py in this folder.

More info:
The ELMo code concatenates the forward and backward LM states at each layer
to form a 1024 dimensional vector. The forward states are the first 512 positions
and the backward states the last 512.  To compute the probability distribution for the
forward LM take the first 512 positions from the top LSTM layer, multiply by W,
add the bias b and apply the softmax.  The backward probabilities are computed in a
similar manner.  If you want, you can verify your implementation by computing the perplexity
of on the held out portion of the 1B Word Benchmark - it should be ~39.

One point of caution about the model implementation due to low level details:
the LM is stateful and the final hidden states are carried over from sentence to sentence.
This has two practical implications:

    Run a few batches through the model to warm up the states before making predictions.
    You will notice this if you compute the perplexity for the first vs subsequent batches
    as the first batch perplexity will be much higher then expected.

    Second, since the LM was trained on randomly shuffled sentences, it relies very heavily
    on the beginning and end of sentence tokens <S> and </S> to reset the internal states.
    Depending on whether you are using the allennlp or tensorflow version of the code the
    batcher may or may not automatically add these tokens (the tensorflow batcher will
    automatically add these tokens, but the allennlp one will not -- they are added
    elsewhere in the code).  As a result, it may require some care to ensure sure your data
    includes the appropriate beginning and end of sentence tokens before running the biLM.

.. code-block:: bash

   $ allennlp elmo --help
   usage: allennlp [command] elmo [-h] [--vocab-path VOCAB_PATH]
                                       [--options-file OPTIONS_FILE]
                                       [--weight-file WEIGHT_FILE]
                                       [--batch-size BATCH_SIZE]
                                       [--cuda-device CUDA_DEVICE]
                                       input_file output_file

   Create perplexities using ELMo.

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
"""

import logging
from typing import IO, List, Iterable, Tuple

import argparse
import h5py
import numpy
import torch

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.modules.elmo import _ElmoBiLm
from allennlp.commands.subcommand import Subcommand
from allennlp.common.file_utils import cached_path
from torch.nn import functional as F

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"  # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"  # pylint: disable=line-too-long
DEFAULT_SOFTMAX_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5"  # pylint: disable=line-too-long
DEFAULT_VOCAB_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/vocab-2016-09-10.txt"  # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64


class ElmoPerplexities(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Compute perplexity using ELMo.'''
        subparser = parser.add_parser(
            name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('input_file', type=argparse.FileType('r'), help='The path to the input file.')
        subparser.add_argument('output_file', type=str, help='The path to the output file.')

        subparser.add_argument('--vocab-path', type=str, default=DEFAULT_VOCAB_FILE,
                               help='The path to the ELMo vocab file.')
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
        subparser.add_argument(
            '--softmax-file',
            type=str,
            default=DEFAULT_SOFTMAX_FILE,
            help='The path to the ELMo softmax weights file.')
        subparser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='The batch size to use.')
        subparser.add_argument('--cuda-device', type=int, default=-1, help='The cuda_device to run on.')

        subparser.set_defaults(func=elmo_command)

        return subparser

class ElmoPerplexityModel:
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 softmax_weight_file: str = DEFAULT_SOFTMAX_FILE,
                 vocab_file: str = DEFAULT_VOCAB_FILE,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        softmax_weight_file : ``str``, optional
            A path or URL to an ELMo softmax weights file.
        vocab_file : ``str``, optional
            A path or URL to an ELMo vocab file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        logger.info("Initializing ELMo.")
        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        self.cuda_device = cuda_device

        self.vocab = Vocabulary()
        self.vocab._oov_token = '<UNK>'
        self.vocab.set_from_file(cached_path(vocab_file), is_padded=False)
        self.fc_layer = torch.nn.Linear(512, self.vocab.get_vocab_size())

        with h5py.File(cached_path(softmax_weight_file), 'r') as fin:
            self.fc_layer.weight.data.copy_(torch.FloatTensor(numpy.array(fin['softmax']['W'])))
            self.fc_layer.bias.data.copy_(torch.FloatTensor(numpy.array(fin['softmax']['b'])))

        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)
            self.fc_layer = self.fc_layer.cuda(device=cuda_device)

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
                              {'character_ids': self.indexer,
                               'word_ids': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=False)})
            instance = Instance({"elmo": field})
            instances.append(instance)

        dataset = Batch(instances)
        dataset.index_instances(self.vocab)
        tensor_dict = dataset.as_tensor_dict(for_training=False)['elmo']
        return tensor_dict['character_ids'], tensor_dict['word_ids']

    def _chunked_log_probs(self, activation, word_targets, chunk_size=256):
        """
        do the softmax in chunks so the gpu ram doesnt explode
        :param activation: [batch, T, dim]
        :param targets: [batch, T] indices
        :param chunk_size: you might need to tune this based on GPU specs
        :return:
        """
        all_logprobs = []
        num_chunks = (activation.size(0) - 1) // chunk_size + 1
        for activation_chunk, target_chunk in zip(torch.chunk(activation, num_chunks, dim=0),
                                                  torch.chunk(word_targets, num_chunks, dim=0)):
            assert activation_chunk.size()[:2] == target_chunk.size()[:2]
            targets_flat = target_chunk.view(-1)
            time_indexer = torch.arange(0, targets_flat.size(0),
                                        out=target_chunk.data.new(targets_flat.size(0))) % target_chunk.size(1)
            batch_indexer = torch.arange(0, targets_flat.size(0),
                                         out=target_chunk.data.new(targets_flat.size(0))) / target_chunk.size(1)
            all_logprobs.append(F.log_softmax(self.fc_layer(activation_chunk), 2)[
                                    batch_indexer, time_indexer, targets_flat].view(*target_chunk.size()))
        return torch.cat(all_logprobs, 0)

    def batch_to_preds(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A tuple of tensors, the first representing perplexities (batch_size, 2, num_timesteps) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids, word_ids = self.batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device, async=True)
            word_ids = word_ids.cuda(device=self.cuda_device, async=True)

        bilm_output = self.elmo_bilm(character_ids)

        activations = bilm_output['activations'][-1]  # (batch, BOS + L + EOS, [fwd, bwd])

        # get rid of predicting, or conditioning on EOS for the forward activation, and BOS for the reverse activation.
        mask_without_bos_eos = bilm_output['mask'][:, 2:]
        fwd_perplexities = self._chunked_log_probs(activations[:, :-2, :512], word_ids)
        bwd_perplexities = self._chunked_log_probs(activations[:, 2:, 512:], word_ids)

        # [B, (fwd, bwd), L]
        perplexities = torch.stack((fwd_perplexities, bwd_perplexities), 1) * mask_without_bos_eos[:, None].float()

        return perplexities, mask_without_bos_eos

    def predict_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        """
        Computes the ELMo perplexities for a batch of tokenized sentences.

        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A list of tensors, each representing the ELMo perplexities for the input sentence at the same index.
        """
        if any([len(s) == 0 for s in batch]):
            raise ValueError("empty sentence found!")

        perplexities, mask = self.batch_to_preds(batch)
        perplexities_np = perplexities.data.cpu().numpy()
        mask_np = mask.data.cpu().numpy()
        return [perp_i[:, :len_i] for perp_i, len_i in zip(perplexities_np, mask_np.sum(1))]

    def predict_sentences(self,
                        sentences: Iterable[List[str]],
                        batch_size: int = DEFAULT_BATCH_SIZE) -> Iterable[numpy.ndarray]:
        """
        Computes the ELMo preddings for a iterable of sentences.

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
            yield from self.predict_batch(batch)

    def predict_file(self,
                   input_file: IO,
                   output_file_path: str,
                   batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """
        Computes ELMo perplexities from an input_file where each line contains a sentence tokenized by whitespace.
        The ELMo perplexities are written out in HDF5 format, where each sentence is saved in a dataset.

        NOTE! the first few perplexities will probably be high because it's

        Parameters
        ----------
        input_file : ``IO``, required
            A file with one tokenized sentence per line.
        output_file_path : ``str``, required
            A path to the output hdf5 file.
        output_format : ``str``, optional, (default = "all")
            The perplexities to output.
        batch_size : ``int``, optional, (default = 64)
            The number of sentences to process in ELMo at one time.
        """
        # Tokenizes the sentences.
        sentences = [line.strip() for line in input_file if line.strip()]
        split_sentences = [sentence.split() for sentence in sentences]
        # Uses the sentence as the key.
        perplexities = zip(sentences, self.predict_sentences(split_sentences, batch_size))

        logger.info("Processing sentences.")
        with h5py.File(output_file_path, 'w') as fout:
            for key, embeddings in Tqdm.tqdm(perplexities):
                fout.create_dataset(
                    key,
                    embeddings.shape, dtype='float32',
                    data=embeddings
                )
        input_file.close()


def elmo_command(args):
    elmo_model = ElmoPerplexityModel(args.options_file, args.weight_file, args.cuda_device)
    elmo_model.predict_file(
        args.input_file,
        args.output_file,
        args.batch_size)

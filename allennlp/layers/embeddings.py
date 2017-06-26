import codecs
import gzip
import logging

import numpy
import torch

from ..data.vocabulary import Vocabulary
from ..common.checks import ConfigurationError
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Embedding(torch.nn.Module):
    """
    A more featureful embedding module than the default in Pytorch.
    Adds the ability to pre-specify the weight matrix, use a non-trainable embedding and
    catches some ill-advised use cases, such as 1 dimensional embeddings.

    Parameters
    ----------
    num_embeddings:, int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim: int
        The size of each embedding vector.
    weight: torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index: int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    max_norm: float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type: float, (optional, default=2):
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq: boolean, (optional, default=False):
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse: bool, (optional, default=False):
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.

    Returns
    -------
    An Embedding module.

    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 weight: torch.FloatTensor=None,
                 padding_index: int=None,
                 trainable: bool=True,
                 max_norm: float=None,
                 norm_type: float=2.,
                 scale_grad_by_freq: bool=False,
                 sparse: bool=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if embedding_dim == 1:
            raise ConfigurationError("There is no need to embed tokens if you are using a 1 dimensional embedding.")

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            self.reset_parameters()

        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            if self.padding_index is not None:
                self.weight.data[self.padding_index].fill_(0)

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding(padding_idx,
                                       self.max_norm,
                                       self.norm_type,
                                       self.scale_grad_by_freq,
                                       self.sparse)(input, self.weight)


def get_pretrained_embedding_layer(embeddings_filename: str,
                                   vocab: Vocabulary,
                                   namespace: str="tokens",
                                   trainable: bool=True,
                                   log_misses: bool=False):
    """
    Reads a pre-trained embedding file and generates a Keras Embedding layer that has weights
    initialized to the pre-trained embeddings.  The Embedding layer can either be trainable or
    not.

    We use the DataIndexer to map from the word strings in the embeddings file to the indices
    that we need, and to know which words from the embeddings file we can safely ignore.  If we
    come across a word in DataIndexer that does not show up with the embeddings file, we give
    it a zero vector.

    The embeddings file is assumed to be gzipped, formatted as [word] [dim 1] [dim 2] ...
    """
    words_to_keep = set(vocab.tokens_in_namespace(namespace))
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}
    embedding_dim = None

    # TODO(matt): make this a parameter
    embedding_misses_filename = 'embedding_misses.txt'

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading embeddings from file")
    with gzip.open(embeddings_filename, 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf-8').strip().split(' ')
            if embedding_dim is None:
                embedding_dim = len(fields) - 1
                assert embedding_dim > 1, "Found embedding size of 1; do you have a header?"
            else:
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    continue
            word = fields[0]
            if word in words_to_keep:
                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[word] = vector

    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    if log_misses:
        logger.info("Logging embedding misses to %s", embedding_misses_filename)
        embedding_misses_file = codecs.open(embedding_misses_filename, 'w', 'utf-8')
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).uniform_(-0.05, 0.05)

    # Depending on whether the namespace has PAD and UNK tokens, we start at different indices,
    # because you shouldn't have pretrained embeddings for PAD or UNK.
    start_index = 0 if namespace.startswith("*") else 2
    for i in range(start_index, vocab_size):
        word = vocab.get_token_from_index(i, namespace)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if word in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[word])
        elif log_misses:
            print(word, file=embedding_misses_file)

    if log_misses:
        embedding_misses_file.close()

    # The weight matrix is initialized, so we construct and return the actual Embedding.
    return Embedding(num_embeddings=vocab_size,
                     embedding_dim=embedding_dim,
                     padding_index=0,
                     weight=embedding_matrix,
                     trainable=trainable)

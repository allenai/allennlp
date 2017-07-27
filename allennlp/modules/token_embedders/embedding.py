import gzip
import logging

from overrides import overrides
import numpy
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TokenEmbedder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("embedding")
class Embedding(TokenEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to
    pre-specify the weight matrix or use a non-trainable embedding.

    Note that if you are using our data API and are trying to embed a
    :class:`~allennlp.data.fields.TextField`, you should use a
    :class:`~allennlp.modules.TextFieldEmbedder` instead of using this directly.

    Parameters
    ----------
    num_embeddings :, int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2):
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False):
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False):
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.

    Returns
    -------
    An Embedding module.

    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            self.weight.data.normal_(0, 1)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

    @overrides
    def get_output_dim(self) -> int:
        return self.embedding_dim

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        padding_index = self.padding_index if self.padding_index is not None else -1
        return self._backend.Embedding(padding_index,
                                       self.max_norm,
                                       self.norm_type,
                                       self.scale_grad_by_freq,
                                       self.sparse)(inputs, self.weight)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        pretrained_file = params.pop("pretrained_file", None)
        if pretrained_file:
            trainable = params.pop("trainable", True)
            return get_pretrained_embedding_layer(pretrained_file, vocab, vocab_namespace, trainable)
        num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop('embedding_dim')
        padding_index = params.pop('padding_index', None)
        trainable = params.pop('trainable', True)
        max_norm = params.pop('max_norm', None)
        norm_type = params.pop('norm_type', 2.)
        scale_grad_by_freq = params.pop('scale_grad_by_freq', False)
        sparse = params.pop('sparse', False)
        params.assert_empty(cls.__name__)
        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse)


def get_pretrained_embedding_layer(embeddings_filename: str,
                                   vocab: Vocabulary,
                                   namespace: str = "tokens",
                                   trainable: bool = True):
    """
    Reads a pre-trained embedding file and generates an Embedding layer that has weights
    initialized to the pre-trained embeddings.  The Embedding layer can either be trainable or
    not.

    We use the ``Vocabulary`` to map from the word strings in the embeddings file to the indices
    that we need, and to know which words from the embeddings file we can safely ignore.

    Parameters
    ----------

    embeddings_filename : str, required.
        The path to a file containing pretrined embeddings. The embeddings
        file is assumed to be gzipped and space delimited, e.g. [word] [dim 1] [dim 2] ...
    vocab : Vocabulary, required.
        A Vocabulary object.
    namespace : str, (optional, default=tokens)
        The namespace of the vocabulary to find pretrained embeddings for.
    trainable : bool, (optional, default=True)
        Whether or not the embedding parameters should be optimized.

    Returns
    -------

    An Embedding Module initialised with a weight matrix of shape
    (vocab.get_vocab_size(namespace), pretrained_embedding_dim),
    where the indices of words appearing in the pretrained embedding file
    are initialized to the pretrained embedding value.

    """
    words_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}
    embedding_dim = None

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
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(0, 1)

    for i in range(0, vocab_size):
        word = vocab.get_token_from_index(i, namespace)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if word in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[word])
        else:
            logger.debug("Word %s was not found in the embedding file. Initialising randomly.", word)

    # The weight matrix is initialized, so we construct and return the actual Embedding.
    return Embedding(num_embeddings=vocab_size,
                     embedding_dim=embedding_dim,
                     padding_index=0,
                     weight=embedding_matrix,
                     trainable=trainable)

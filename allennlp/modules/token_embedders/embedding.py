import gzip
import logging

from overrides import overrides
import numpy
import torch
from torch.nn.functional import embedding
import h5py

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("embedding")
class Embedding(TokenEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
        5. build all of this easily ``from_params``

    Note that if you are using our data API and are trying to embed a
    :class:`~allennlp.data.fields.TextField`, you should use a
    :class:`~allennlp.modules.TextFieldEmbedder` instead of using this directly.

    Parameters
    ----------
    num_embeddings :, int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    projection_dim : int, (optional, default=None)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if ``trainable`` is ``False``.
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
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.output_dim = projection_dim or embedding_dim

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

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        embedded = embedding(inputs, self.weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)
        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()) + [embedded.size(-1)]
            embedded = embedded.view(*view_args)
        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
        key directly, and the vocabulary will be ignored.
        """
        num_embeddings = params.pop_int('num_embeddings', None)
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop("pretrained_file", None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        params.assert_empty(cls.__name__)

        if pretrained_file:
            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.
            weight = _read_pretrained_embedding_file(pretrained_file,
                                                     embedding_dim,
                                                     vocab,
                                                     vocab_namespace)
        else:
            weight = None

        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   weight=weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse)


def _read_pretrained_embedding_file(embeddings_filename: str,
                                    embedding_dim: int,
                                    vocab: Vocabulary,
                                    namespace: str = "tokens") -> torch.FloatTensor:
    """
    Reads a pre-trained embedding file and generates an Embedding layer that has weights
    initialized to the pre-trained embeddings.  The Embedding layer can either be trainable or
    not.

    We use the ``Vocabulary`` to map from the word strings in the embeddings file to the indices
    that we need, and to know which words from the embeddings file we can safely ignore.

    Parameters
    ----------
    embeddings_filename : str, required.
        The path to a file containing pretrained embeddings. We support two file formats,
        gzipped-word2vec and hdf5.  If the filename ends with '.hdf5' or '.h5' then we load from
        hdf5, otherwise assume gzipped-word2vec format.
    vocab : Vocabulary, required.
        A Vocabulary object.
    namespace : str, (optional, default=tokens)
        The namespace of the vocabulary to find pretrained embeddings for.
    trainable : bool, (optional, default=True)
        Whether or not the embedding parameters should be optimized.

    Returns
    -------
    A weight matrix with embeddings initialized from the read file.  The matrix has shape
    ``(vocab.get_vocab_size(namespace), embedding_dim)``, where the indices of words appearing in
    the pretrained embedding file are initialized to the pretrained embedding value.
    """
    if embeddings_filename[-3:] == '.h5' or embeddings_filename[-5:] == '.hdf5':
        return _read_pretrained_hdf5_format_embedding_file(embeddings_filename, embedding_dim,
                                                           vocab, namespace)
    else:
        # default to word2vec
        return _read_pretrained_word2vec_format_embedding_file(embeddings_filename, embedding_dim,
                                                               vocab, namespace)


def _read_pretrained_word2vec_format_embedding_file(embeddings_filename: str, # pylint: disable=invalid-name
                                                    embedding_dim: int,
                                                    vocab: Vocabulary,
                                                    namespace: str = "tokens") -> torch.FloatTensor:
    """
    Read from a gzipped-word2vec format file.  The embeddings file is assumed to be gzipped and
    space delimited, e.g. [word] [dim 1] [dim 2] ...

    The remainder of the docstring is identical to ``_read_pretrained_embedding_file``.
    """
    words_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading embeddings from file")
    with gzip.open(cached_path(embeddings_filename), 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf-8').strip().split(' ')
            if len(fields) - 1 != embedding_dim:
                # Sometimes there are funny unicode parsing problems that lead to different
                # fields lengths (e.g., a word with a unicode space character that splits
                # into more than one column).  We skip those lines.  Note that if you have
                # some kind of long header, this could result in all of your lines getting
                # skipped.  It's hard to check for that here; you just have to look in the
                # embedding_misses_file and at the model summary to make sure things look
                # like they are supposed to.
                logger.warning("Found line with wrong number of dimensions (expected %d, was %d): %s",
                               embedding_dim, len(fields) - 1, line)
                continue
            word = fields[0]
            if word in words_to_keep:
                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[word] = vector

    if not embeddings:
        raise ConfigurationError("No embeddings of correct dimension found; you probably "
                                 "misspecified your embedding_dim parameter, or didn't "
                                 "pre-populate your Vocabulary")

    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)

    for i in range(0, vocab_size):
        word = vocab.get_token_from_index(i, namespace)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if word in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[word])
        else:
            logger.debug("Word %s was not found in the embedding file. Initialising randomly.", word)

    # The weight matrix is initialized, so we construct and return the actual Embedding.
    return embedding_matrix


def _read_pretrained_hdf5_format_embedding_file(embeddings_filename: str, # pylint: disable=invalid-name
                                                embedding_dim: int,
                                                vocab: Vocabulary,
                                                namespace: str = "tokens") -> torch.FloatTensor:
    """
    Reads from a hdf5 formatted file.  The embedding matrix is assumed to
    be keyed by 'embedding' and of size ``(num_tokens, embedding_dim)``.
    """
    with h5py.File(embeddings_filename, 'r') as fin:
        embeddings = fin['embedding'][...]

    if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
        raise ConfigurationError(
                "Read shape {0} embeddings from the file, but expected {1}".format(
                        list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]))

    return torch.FloatTensor(embeddings)

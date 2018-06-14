import io
import tarfile
import zipfile
import bz2
import lzma
import gzip
import logging
import warnings
from contextlib import contextmanager
from typing import Optional, ContextManager, Tuple, Union, Sequence, cast, IO, TextIO

from overrides import overrides
import numpy
import torch
from torch.nn.functional import embedding

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import get_file_extension, cached_path
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


EMBEDDINGS_FILE_ENCODING = 'utf-8'


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
            torch.nn.init.xavier_uniform_(self.weight)
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

        A file containing pretrained embeddings can be specified using the parameter ``pretrained_file``.
        Two formats are supported: hdf5 and text file.
        The text file is assumed to be utf-8 encoded and space separated: [word] [dim 1] [dim 2] ...
        The text file can eventually be compressed with gzip, bz2, lzma or zip. Furthermore,
        it can also resides inside a zip or tar archive with multiple files. In this last case,
        the parameter ``pretrained_file`` must be a pair ``[archive_path, path_inside_archive]``.
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
            weight = _read_pretrained_embeddings_file(pretrained_file,
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


def _read_pretrained_embeddings_file(embeddings_filename: Union[str, Sequence[str]],
                                     embedding_dim: int,
                                     vocab: Vocabulary,
                                     namespace: str = "tokens") -> torch.FloatTensor:
    """
    Returns and embedding matrix for the given vocabulary using the pretrained embeddings
    contained in the given file. Embeddings for tokens not found in the pretrained embedding file
    are randomly initialized using a normal distribution with mean and standard deviation equal to
    those of the pretrained embeddings.

    We support two file formats:

        * text format - utf-8 encoded text file with space separated fields: [word] [dim 1] [dim 2] ...
          The text file can eventually be compressed, and even resides in an archive with multiple files.
          If the file resides in an archive with other files, then ``embeddings_filename`` must
          be a pair of strings ``[archive_path, path_inside_archive]``

        * hdf5 format - hdf5 file containing an embedding matrix in the form of a torch.Tensor.

    If the filename ends with '.hdf5' or '.h5' then we load from hdf5, otherwise we assume
    text format.

    Parameters
    ----------
    embeddings_filename : Union[str, Sequence[str]], required.
        Path to the file containing the embeddings. It can be a string or a pair of strings.
        If the file is an (eventually compressed) text file or a file contained in a zip/tar
        archive containing only a single file, a string is enough. If otherwise the file resides
        in an archive with multiple other files, a pair [archive_path, path_inside_archive] is
        needed.
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
    if isinstance(embeddings_filename, str):
        file_ext = get_file_extension(embeddings_filename)
        if file_ext in ['.h5', '.hdf5']:
            return _read_embeddings_from_hdf5(embeddings_filename,
                                              embedding_dim,
                                              vocab, namespace)

    return _read_embeddings_from_text_file(embeddings_filename,
                                           embedding_dim,
                                           vocab, namespace)


def _get_the_only_file_in_the_archive(members_list: Sequence[str],
                                      archive_path: str):
    if len(members_list) > 1:
        raise ValueError('The archive %s contains multiple files, so you must select '
                         'one of the files inside it providing a pair '
                         '[archive_path, path_of_the_file_inside_archive]' % archive_path)
    return members_list[0]


def normalize_embeddings_filename(embeddings_filename: Union[str, Sequence[str]]) -> Tuple[str, Optional[str]]:
    """
    If embedding_filename is a string, returns (embedding_filename, '').
    If it's a pair of string, it returns them in a tuple
    """
    if isinstance(embeddings_filename, Sequence) and len(embeddings_filename) == 2:
        return str(embeddings_filename[0]), str(embeddings_filename[1])
    elif isinstance(embeddings_filename, str):
        return str(embeddings_filename), None
    else:
        raise ValueError('Invalid path to pretrained embeddings: %r\n'
                         'It must be a string or a pair or strings [archive_path, member_path]'
                         % embeddings_filename)


@contextmanager  # type: ignore
def open_embeddings_text_file(embeddings_filename: Union[str, Sequence[str]],
                              encoding: str = EMBEDDINGS_FILE_ENCODING,
                              cache_dir: str = None) -> ContextManager[TextIO]:
    """
    Utility function for opening embeddings text files. The file can be
        * a plain uncompressed text file
        * a text file compressed with zip, gzip, bz2 or lzma
        * a text file in a zip or tar archive containing multiple files; in this case,
          the argument ``embeddings_filename`` must be a pair [archive_path, path_inside_archive]
    """
    first_level_path, second_level_path = normalize_embeddings_filename(embeddings_filename)
    cached_first_level_path = cached_path(first_level_path, cache_dir=cache_dir)

    if zipfile.is_zipfile(cached_first_level_path):  # ZIP archive
        logger.debug('Reading pretrained embeddings file contained in a zip archive')
        with zipfile.ZipFile(cached_first_level_path, 'r') as zip_archive:
            if second_level_path is None:
                members_list = zip_archive.namelist()
                second_level_path = _get_the_only_file_in_the_archive(members_list, first_level_path)
            second_level_path = cast(str, second_level_path)   # mypy is not smart enough
            with io.TextIOWrapper(zip_archive.open(second_level_path, 'r'),
                                  encoding=encoding) as embeddings_file:
                yield embeddings_file

    elif tarfile.is_tarfile(first_level_path):  # TAR archive
        logger.debug('Reading pretrained embeddings file contained in a tar archive')
        with tarfile.open(cached_first_level_path, 'r') as tar_archive:
            if second_level_path is None:
                members_list = tar_archive.getnames()
                second_level_path = _get_the_only_file_in_the_archive(members_list, first_level_path)
            second_level_path = cast(str, second_level_path)  # mypy is not smart enough
            tar_member = tar_archive.getmember(second_level_path)
            tar_member_file = tar_archive.extractfile(tar_member)
            if not tar_member_file:
                raise ValueError('File %s not found in the archive %s' %
                                 (second_level_path, first_level_path))
            tar_member_file = cast(IO[bytes], tar_member_file)

            with io.TextIOWrapper(tar_member_file, encoding=encoding) as embeddings_file:
                yield embeddings_file

    else:  # (eventually compressed) text file
        if second_level_path:
            raise ValueError('Unsupported archive format: %s' + first_level_path)

        # All the python packages for compressed files share the same interface of io.open
        extension = get_file_extension(first_level_path)
        package = {
                '.txt': io,
                '.vec': io,
                '.gz': gzip,
                '.bz2': bz2,
                '.lzma': lzma,
                }.get(extension, None)

        if package is None:
            logger.warning('The embedding file has an unknown file extension "%s". '
                           'We will assume the file is an (uncompressed) text file', extension)
            package = io

        with package.open(cached_first_level_path, 'rt', encoding=encoding) as embeddings_file:  # type: ignore
            yield embeddings_file


def read_num_pretrained_tokens_if_present(embeddings_filename: Union[str, Sequence[str]]) -> Optional[int]:
    """ Some pretrained embedding files (e.g. FastText) start declaring the number of tokens
    and the embedding size. The former is useful for showing progress. This function read
    the first row and if it contains 1 or 2 integers, it assumes that the biggest one is
    the number of tokens """
    num_tokens = None
    with open_embeddings_text_file(embeddings_filename) as embeddings_file:  # type: TextIO
        first_line = embeddings_file.readline()
        fields = first_line.split(' ')
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except TypeError:
                pass
            else:
                num_tokens = max(int_fields)
    if num_tokens:
        logger.info('Number of pretrained tokens heuristically inferred from the first row: %d', num_tokens)
    return num_tokens


def _read_embeddings_from_text_file(embeddings_filename: Union[str, Sequence[str]],  # pylint: disable=invalid-name
                                    embedding_dim: int,
                                    vocab: Vocabulary,
                                    namespace: str = "tokens") -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    If the file is contained in an archive with other files, then ``embeddings_filename`` must
    be a pair ``[archive_path, path_of_the_file_inside_archive]``.

    Lines that contain more numerical tokens than ``embedding_dim`` raise a warning and are skipped.

    The remainder of the docstring is identical to ``_read_pretrained_embeddings_file``.
    """
    words_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pretrained embeddings from file")

    num_pretrained_tokens = read_num_pretrained_tokens_if_present(embeddings_filename)

    with open_embeddings_text_file(embeddings_filename) as embeddings_file:  # type: TextIO
        if num_pretrained_tokens:
            embeddings_file.readline()  # skip header

        for line in Tqdm.tqdm(embeddings_file, total=num_pretrained_tokens):
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                # Sometimes there are funny unicode parsing problems that lead to different
                # fields lengths (e.g., a word with a unicode space character that splits
                # into more than one column).  We skip those lines.  Note that if you have
                # some kind of long header, this could result in all of your lines getting
                # skipped.  It's hard to check for that here; you just have to look in the
                # embedding_misses_file and at the model summary to make sure things look
                # like they are supposed to.
                logger.warning("Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
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
    num_found_tokens = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        word = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if word in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[word])
            num_found_tokens += 1
        else:
            logger.debug("Word %s was not found in the embedding file. Initialising randomly.", word)

    logger.info("%d out of %d tokens were found in the embedding file", num_found_tokens, vocab_size)
    # The weight matrix is initialized, so we construct and return the actual Embedding.
    return embedding_matrix


def _read_embeddings_from_hdf5(embeddings_filename: str,  # pylint: disable=invalid-name
                               embedding_dim: int,
                               vocab: Vocabulary,
                               namespace: str = "tokens") -> torch.FloatTensor:
    """
    Reads from a hdf5 formatted file. The embedding matrix is assumed to
    be keyed by 'embedding' and of size ``(num_tokens, embedding_dim)``.
    """
    with h5py.File(embeddings_filename, 'r') as fin:
        embeddings = fin['embedding'][...]

    if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
        raise ConfigurationError(
                "Read shape {0} embeddings from the file, but expected {1}".format(
                        list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]))

    return torch.FloatTensor(embeddings)

import io
import tarfile
import zipfile
import bz2
import lzma
import gzip
import re
import logging
import itertools
import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, Sequence, cast, IO, TextIO, Iterator

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
        It can be the path to a local file or an URL to a (cached) remote file.
        Two formats are supported:

            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;

            * text file - an utf-8 encoded text file with space separated fields::

                    [word] [dim 1] [dim 2] ...

              The text file can eventually be compressed with gzip, bz2, lzma or zip.
              You can even select a single file inside an archive containing multiple files
              using the URI::

                    "(archive_uri)#file_path_inside_the_archive"

              where ``archive_uri`` can be a file system path or a URL. For example:

                    "(http://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"
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


def _read_pretrained_embeddings_file(embeddings_file_uri: str,
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
          be a URI "(archive_uri)#file_path_inside_the_archive"

        * hdf5 format - hdf5 file containing an embedding matrix in the form of a torch.Tensor.

    If the filename ends with '.hdf5' or '.h5' then we load from hdf5, otherwise we assume
    text format.

    Parameters
    ----------
    embeddings_file_uri : str, required.
        It can be:

        * a file system path or a URL to an eventually compressed text file or a zip/tar archive
          containing a single file.

        * URI of the type ``(archive_path_or_url)#file_path_inside_archive`` if the text file
          is contained in a multi-file archive.

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
    file_ext = get_file_extension(embeddings_file_uri)
    if file_ext in ['.h5', '.hdf5']:
        return _read_embeddings_from_hdf5(embeddings_file_uri,
                                          embedding_dim,
                                          vocab, namespace)

    return _read_embeddings_from_text_file(embeddings_file_uri,
                                           embedding_dim,
                                           vocab, namespace)


def _get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
    if len(members_list) > 1:
        raise ValueError('The archive %s contains multiple files, so you must select '
                         'one of the files inside it providing a pair '
                         '[archive_path, path_of_the_file_inside_archive]' % archive_path)
    return members_list[0]


def get_embeddings_file_uri(path_or_url: str, path_inside_archive: Optional[str] = None) -> str:
    if path_inside_archive:
        return "({})#{}".format(path_or_url, path_inside_archive)
    return path_or_url


def decode_embeddings_file_uri(uri: str) -> Tuple[str, Optional[str]]:
    match = re.fullmatch('\((.*)\)#(.*)', uri)      # pylint: disable=anomalous-backslash-in-string
    if match:
        return cast(Tuple[str, str], match.groups())
    return uri, None


@contextmanager
def open_embeddings_text_file(embeddings_file_uri: str,
                              encoding: str = EMBEDDINGS_FILE_ENCODING,
                              cache_dir: str = None) -> Iterator[TextIO]:
    """
    Utility function for opening embeddings text files.

    Parameters
    ----------
    embeddings_file_uri: str
        It can be:

        * a file system path or a URL to an eventually compressed text file or a zip/tar archive
          containing a single file.

        * URI of the type ``(archive_path_or_url)#file_path_inside_archive`` if the text file
          is contained in a multi-file archive.

    encoding: str
    cache_dir: str
    """
    first_level_path, second_level_path = decode_embeddings_file_uri(embeddings_file_uri)
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


def _get_num_tokens_in_file_from_1st_line(first_line: str) -> Optional[int]:
    """
    Some pretrained embedding files (e.g. FastText) start declaring the number of tokens
    and the embedding size. The former is useful for showing progress.

    This function takes in input the first line and if it's a "valid" header, it returns
    the number of tokens in the embedding file. It assumes a header is composed of 1 or 2 integers
    and that the maximum one (or the only one) is the number of pretrained tokens.

    It returns None if the string doesn't match this pattern.
    """
    fields = first_line.split(' ')
    if 1 <= len(fields) <= 2:
        try:
            int_fields = [int(x) for x in fields]
        except ValueError:
            return None
        else:
            num_tokens = max(int_fields)
            logger.info('Number of pretrained tokens heuristically inferred from the first row: %d',
                        num_tokens)
            return num_tokens
    return None


def get_embeddings_file_iterator_with_progbar(embeddings_file: TextIO):
    """
    Some pretrained embedding files (e.g. FastText) start with a header containing the number
    of tokens and the size of the vectors. The former is useful for showing progress when reading
    the file.

    This function read the first line of the file to see if it's a header containing the number
    of pretrained tokens in the file; then it returns a file iterator decorated with a progress bar.
    If the first line is a "valid header" (see :func:`_get_num_tokens_in_file_from_1st_line`),
    then the returned iterator starts from the 2nd line of the file and the progress bar is set
    with an expected number of iterations. Otherwise, the returned iterator will start from the 1st
    line and the progress bar will run without an expected number of iterations (better than nothing).
    """
    first_line = next(embeddings_file)
    num_pretrained_tokens = _get_num_tokens_in_file_from_1st_line(first_line)

    if num_pretrained_tokens:
        return Tqdm.tqdm(embeddings_file)  # skip the first line (header)
    else:
        # don't skip the first line
        return Tqdm.tqdm(itertools.chain([first_line], embeddings_file),
                         total=num_pretrained_tokens)


def _read_embeddings_from_text_file(embeddings_file_uri: str,
                                    embedding_dim: int,
                                    vocab: Vocabulary,
                                    namespace: str = "tokens") -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than ``embedding_dim`` raise a warning and are skipped.

    The remainder of the docstring is identical to ``_read_pretrained_embeddings_file``.
    """
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pretrained embeddings from file")

    with open_embeddings_text_file(embeddings_file_uri) as embeddings_file:

        for line in get_embeddings_file_iterator_with_progbar(embeddings_file):
            token = line.split(' ', 1)[0]
            if token in tokens_to_keep:
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

                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

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
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug("Token %s was not found in the embedding file. Initialising randomly.", token)

    logger.info("Pretrained embeddings were found for %d out of %d tokens",
                num_tokens_found, vocab_size)

    return embedding_matrix


def _read_embeddings_from_hdf5(embeddings_filename: str,
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

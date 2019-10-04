import io
import tarfile
import zipfile
import re
import logging
import warnings
import itertools
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple

from overrides import overrides
import numpy
import torch
from torch.nn.functional import embedding

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import get_file_extension, cached_path, is_url_or_existing_file
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

logger = logging.getLogger(__name__)


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
    num_embeddings : int
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
    norm_type : float, (optional, default=2)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False)
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : str, (optional, default=None)
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : str, (optional, default=None)
        Used to keep track of what is the source of the weights and loading more embeddings at test time.
        **It does not load the weights from this pretrained_file.** For that purpose, use
        ``Embedding.from_params``.

    Returns
    -------
    An Embedding module.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        projection_dim: int = None,
        weight: torch.FloatTensor = None,
        padding_index: int = None,
        trainable: bool = True,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        vocab_namespace: str = None,
        pretrained_file: str = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError(
                    "A weight matrix was passed with contradictory embedding shapes."
                )
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
    def forward(self, inputs):
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        embedded = embedding(
            inputs,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    def extend_vocab(
        self,
        extended_vocab: Vocabulary,
        vocab_namespace: str = None,
        extension_pretrained_file: str = None,
        model_path: str = None,
    ):
        """
        Extends the embedding matrix according to the extended vocabulary.
        If extension_pretrained_file is available, it will be used for initializing the new words
        embeddings in the extended vocabulary; otherwise we will check if _pretrained_file attribute
        is already available. If none is available, they will be initialized with xavier uniform.

        Parameters
        ----------
        extended_vocab : Vocabulary:
            Vocabulary extended from original vocabulary used to construct
            this ``Embedding``.
        vocab_namespace : str, (optional, default=None)
            In case you know what vocab_namespace should be used for extension, you
            can pass it. If not passed, it will check if vocab_namespace used at the
            time of ``Embedding`` construction is available. If so, this namespace
            will be used or else extend_vocab will be a no-op.
        extension_pretrained_file : str, (optional, default=None)
            A file containing pretrained embeddings can be specified here. It can be
            the path to a local file or an URL of a (cached) remote file. Check format
            details in ``from_params`` of ``Embedding`` class.
        model_path : str, (optional, default=None)
            Path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens". This is only useful
            to give helpful error message when extend_vocab is implicitly called
            by fine-tune or any other command.
        """
        # Caveat: For allennlp v0.8.1 and below, we weren't storing vocab_namespace as an attribute,
        # knowing which is necessary at time of embedding vocab extension. So old archive models are
        # currently unextendable.

        vocab_namespace = vocab_namespace or self._vocab_namespace
        if not vocab_namespace:
            # It's not safe to default to "tokens" or any other namespace.
            logging.info(
                "Loading a model trained before embedding extension was implemented; "
                "pass an explicit vocab namespace if you want to extend the vocabulary."
            )
            return

        extended_num_embeddings = extended_vocab.get_vocab_size(vocab_namespace)
        if extended_num_embeddings == self.num_embeddings:
            # It's already been extended. No need to initialize / read pretrained file in first place (no-op)
            return

        if extended_num_embeddings < self.num_embeddings:
            raise ConfigurationError(
                f"Size of namespace, {vocab_namespace} for extended_vocab is smaller than "
                f"embedding. You likely passed incorrect vocab or namespace for extension."
            )

        # Case 1: user passed extension_pretrained_file and it's available.
        if extension_pretrained_file and is_url_or_existing_file(extension_pretrained_file):
            # Don't have to do anything here, this is the happy case.
            pass
        # Case 2: user passed extension_pretrained_file and it's not available
        elif extension_pretrained_file:
            raise ConfigurationError(
                f"You passed pretrained embedding file {extension_pretrained_file} "
                f"for model_path {model_path} but it's not available."
            )
        # Case 3: user didn't pass extension_pretrained_file, but pretrained_file attribute was
        # saved during training and is available.
        elif is_url_or_existing_file(self._pretrained_file):
            extension_pretrained_file = self._pretrained_file
        # Case 4: no file is available, hope that pretrained embeddings weren't used in the first place and warn
        else:
            extra_info = (
                f"Originally pretrained_file was at " f"{self._pretrained_file}. "
                if self._pretrained_file
                else ""
            )
            # It's better to warn here and not give error because there is no way to distinguish between
            # whether pretrained-file wasn't used during training or user forgot to pass / passed incorrect
            # mapping. Raising an error would prevent fine-tuning in the former case.
            logging.warning(
                f"Embedding at model_path, {model_path} cannot locate the pretrained_file. "
                f"{extra_info} If you are fine-tuning and want to use using pretrained_file for "
                f"embedding extension, please pass the mapping by --embedding-sources argument."
            )

        embedding_dim = self.weight.data.shape[-1]
        if not extension_pretrained_file:
            extra_num_embeddings = extended_num_embeddings - self.num_embeddings
            extra_weight = torch.FloatTensor(extra_num_embeddings, embedding_dim)
            torch.nn.init.xavier_uniform_(extra_weight)
        else:
            # It's easiest to just reload the embeddings for the entire vocab,
            # then only keep the ones we need.
            whole_weight = _read_pretrained_embeddings_file(
                extension_pretrained_file, embedding_dim, extended_vocab, vocab_namespace
            )
            extra_weight = whole_weight[self.num_embeddings :, :]

        device = self.weight.data.device
        extended_weight = torch.cat([self.weight.data, extra_weight.to(device)], dim=0)
        self.weight = torch.nn.Parameter(extended_weight, requires_grad=self.weight.requires_grad)

    # Custom logic requires custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "Embedding":  # type: ignore
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
        key directly, and the vocabulary will be ignored.

        In the configuration file, a file containing pretrained embeddings can be specified
        using the parameter ``"pretrained_file"``.
        It can be the path to a local file or an URL of a (cached) remote file.
        Two formats are supported:

            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;

            * text file - an utf-8 encoded text file with space separated fields::

                    [word] [dim 1] [dim 2] ...

              The text file can eventually be compressed with gzip, bz2, lzma or zip.
              You can even select a single file inside an archive containing multiple files
              using the URI::

                    "(archive_uri)#file_path_inside_the_archive"

              where ``archive_uri`` can be a file system path or a URL. For example::

                    "(https://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"
        """

        num_embeddings = params.pop_int("num_embeddings", None)
        # If num_embeddings is present, set default namespace to None so that extend_vocab
        # call doesn't misinterpret that some namespace was originally used.
        vocab_namespace = params.pop("vocab_namespace", None if num_embeddings else "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int("embedding_dim")
        pretrained_file = params.pop("pretrained_file", None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int("padding_index", None)
        max_norm = params.pop_float("max_norm", None)
        norm_type = params.pop_float("norm_type", 2.0)
        scale_grad_by_freq = params.pop_bool("scale_grad_by_freq", False)
        sparse = params.pop_bool("sparse", False)
        params.assert_empty(cls.__name__)

        if pretrained_file:
            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.
            weight = _read_pretrained_embeddings_file(
                pretrained_file, embedding_dim, vocab, vocab_namespace
            )
        else:
            weight = None

        return cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            weight=weight,
            padding_index=padding_index,
            trainable=trainable,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            vocab_namespace=vocab_namespace,
        )


def _read_pretrained_embeddings_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
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
    file_uri : str, required.
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
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
    file_ext = get_file_extension(file_uri)
    if file_ext in [".h5", ".hdf5"]:
        return _read_embeddings_from_hdf5(file_uri, embedding_dim, vocab, namespace)

    return _read_embeddings_from_text_file(file_uri, embedding_dim, vocab, namespace)


def _read_embeddings_from_text_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
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

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(" ")
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning(
                        "Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                        embedding_dim,
                        len(fields) - 1,
                        line,
                    )
                    continue

                vector = numpy.asarray(fields[1:], dtype="float32")
                embeddings[token] = vector

    if not embeddings:
        raise ConfigurationError(
            "No embeddings of correct dimension found; you probably "
            "misspecified your embedding_dim parameter, or didn't "
            "pre-populate your Vocabulary"
        )

    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(
        embeddings_mean, embeddings_std
    )
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
            logger.debug(
                "Token %s was not found in the embedding file. Initialising randomly.", token
            )

    logger.info(
        "Pretrained embeddings were found for %d out of %d tokens", num_tokens_found, vocab_size
    )

    return embedding_matrix


def _read_embeddings_from_hdf5(
    embeddings_filename: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    """
    Reads from a hdf5 formatted file. The embedding matrix is assumed to
    be keyed by 'embedding' and of size ``(num_tokens, embedding_dim)``.
    """
    with h5py.File(embeddings_filename, "r") as fin:
        embeddings = fin["embedding"][...]

    if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
        raise ConfigurationError(
            "Read shape {0} embeddings from the file, but expected {1}".format(
                list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]
            )
        )

    return torch.FloatTensor(embeddings)


def format_embeddings_file_uri(
    main_file_path_or_url: str, path_inside_archive: Optional[str] = None
) -> str:
    if path_inside_archive:
        return "({})#{}".format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


class EmbeddingsFileURI(NamedTuple):
    main_file_uri: str
    path_inside_archive: Optional[str] = None


def parse_embeddings_file_uri(uri: str) -> "EmbeddingsFileURI":
    match = re.fullmatch(r"\((.*)\)#(.*)", uri)
    if match:
        fields = cast(Tuple[str, str], match.groups())
        return EmbeddingsFileURI(*fields)
    else:
        return EmbeddingsFileURI(uri, None)


class EmbeddingsTextFile(Iterator[str]):
    """
    Utility class for opening embeddings text files. Handles various compression formats,
    as well as context management.

    Parameters
    ----------
    file_uri: str
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
          containing a single file.
        * URI of the type ``(archive_path_or_url)#file_path_inside_archive`` if the text file
          is contained in a multi-file archive.

    encoding: str
    cache_dir: str
    """

    DEFAULT_ENCODING = "utf-8"

    def __init__(
        self, file_uri: str, encoding: str = DEFAULT_ENCODING, cache_dir: str = None
    ) -> None:

        self.uri = file_uri
        self._encoding = encoding
        self._cache_dir = cache_dir
        self._archive_handle: Any = None  # only if the file is inside an archive

        main_file_uri, path_inside_archive = parse_embeddings_file_uri(file_uri)
        main_file_local_path = cached_path(main_file_uri, cache_dir=cache_dir)

        if zipfile.is_zipfile(main_file_local_path):  # ZIP archive
            self._open_inside_zip(main_file_uri, path_inside_archive)

        elif tarfile.is_tarfile(main_file_local_path):  # TAR archive
            self._open_inside_tar(main_file_uri, path_inside_archive)

        else:  # all the other supported formats, including uncompressed files
            if path_inside_archive:
                raise ValueError("Unsupported archive format: %s" + main_file_uri)

            # All the python packages for compressed files share the same interface of io.open
            extension = get_file_extension(main_file_uri)

            # Some systems don't have support for all of these libraries, so we import them only
            # when necessary.
            package = None
            if extension in [".txt", ".vec"]:
                package = io
            elif extension == ".gz":
                import gzip

                package = gzip
            elif extension == ".bz2":
                import bz2

                package = bz2
            elif extension == ".lzma":
                import lzma

                package = lzma

            if package is None:
                logger.warning(
                    'The embeddings file has an unknown file extension "%s". '
                    "We will assume the file is an (uncompressed) text file",
                    extension,
                )
                package = io

            self._handle = package.open(  # type: ignore
                main_file_local_path, "rt", encoding=encoding
            )

        # To use this with tqdm we'd like to know the number of tokens. It's possible that the
        # first line of the embeddings file contains this: if it does, we want to start iteration
        # from the 2nd line, otherwise we want to start from the 1st.
        # Unfortunately, once we read the first line, we cannot move back the file iterator
        # because the underlying file may be "not seekable"; we use itertools.chain instead.
        first_line = next(self._handle)  # this moves the iterator forward
        self.num_tokens = EmbeddingsTextFile._get_num_tokens_from_first_line(first_line)
        if self.num_tokens:
            # the first line is a header line: start iterating from the 2nd line
            self._iterator = self._handle
        else:
            # the first line is not a header line: start iterating from the 1st line
            self._iterator = itertools.chain([first_line], self._handle)

    def _open_inside_zip(self, archive_path: str, member_path: Optional[str] = None) -> None:
        cached_archive_path = cached_path(archive_path, cache_dir=self._cache_dir)
        archive = zipfile.ZipFile(cached_archive_path, "r")
        if member_path is None:
            members_list = archive.namelist()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member_file = archive.open(member_path, "r")
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def _open_inside_tar(self, archive_path: str, member_path: Optional[str] = None) -> None:
        cached_archive_path = cached_path(archive_path, cache_dir=self._cache_dir)
        archive = tarfile.open(cached_archive_path, "r")
        if member_path is None:
            members_list = archive.getnames()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member = archive.getmember(member_path)  # raises exception if not present
        member_file = cast(IO[bytes], archive.extractfile(member))
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def read(self) -> str:
        return "".join(self._iterator)

    def readline(self) -> str:
        return next(self._iterator)

    def close(self) -> None:
        self._handle.close()
        if self._archive_handle:
            self._archive_handle.close()

    def __enter__(self) -> "EmbeddingsTextFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> "EmbeddingsTextFile":
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def __len__(self) -> Optional[int]:
        """ Hack for tqdm: no need for explicitly passing ``total=file.num_tokens`` """
        if self.num_tokens:
            return self.num_tokens
        raise AttributeError(
            'an object of type EmbeddingsTextFile has "len()" only if the underlying '
            "text file declares the number of tokens (i.e. the number of lines following)"
            "in the first line. That is not the case of this particular instance."
        )

    @staticmethod
    def _get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
        if len(members_list) > 1:
            raise ValueError(
                "The archive %s contains multiple files, so you must select "
                "one of the files inside providing a uri of the type: %s."
                % (
                    archive_path,
                    format_embeddings_file_uri("path_or_url_to_archive", "path_inside_archive"),
                )
            )
        return members_list[0]

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) -> Optional[int]:
        """ This function takes in input a string and if it contains 1 or 2 integers, it assumes the
        largest one it the number of tokens. Returns None if the line doesn't match that pattern. """
        fields = line.split(" ")
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except ValueError:
                return None
            else:
                num_tokens = max(int_fields)
                logger.info(
                    "Recognized a header line in the embedding file with number of tokens: %d",
                    num_tokens,
                )
                return num_tokens
        return None

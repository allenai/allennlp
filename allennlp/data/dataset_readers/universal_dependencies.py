from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def lazy_parse(text: str, fields: Tuple = DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]


@DatasetReader.register("universal_dependencies")
class UniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format. Additionally,
    in order to make it easy to structure a model as predicting arcs, we add a
    dummy 'ROOT_HEAD' token to the start of the sequence.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_pos_tags : ``bool``, optional, (default = ``False``)
        Whether or not the instance should contain gold POS tags
        as a field.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_pos_tags: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._use_pos_tags = use_pos_tags
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in  lazy_parse(conllu_file.read()):

                yield self.text_to_instance(
                        [x["form"] for x in annotation],
                        [x["upostag"] for x in annotation] if self._use_pos_tags else None,
                        [x["deps"][0] for x in annotation])

    @overrides
    def text_to_instance(self,  # type: ignore
                         words: List[str],
                         upos_tags: List[str] = None,
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, optional (default = None).
            The universal dependencies POS tags for each word.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        # In order to make it easy to structure a model as predicting arcs, we add a
        # dummy 'ROOT_HEAD' token to the start of the sequence. This will be masked in the
        # loss function.
        tokens = TextField([Token("ROOT_HEAD")] + [Token(w) for w in words], self._token_indexers)
        fields["words"] = tokens
        if self._use_pos_tags and upos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(["ROOT_POS"] + upos_tags, tokens, label_namespace="pos")
        # We don't want to expand the label namespace with an additional dummy token, so we'll
        # always give the 'ROOT_HEAD' token a label of 'root'.
        fields["head_tags"] = SequenceLabelField(["root"] + [x[0] for x in dependencies],
                                                 tokens,
                                                 label_namespace="head_tags")
        if dependencies is not None:
            fields["head_indices"] = SequenceLabelField([0] + [int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")
        return Instance(fields)

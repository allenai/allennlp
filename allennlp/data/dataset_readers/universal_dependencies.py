from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def lazy_parse(text: str, fields: Tuple = DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")
                  ]


@DatasetReader.register("universal_dependencies")
class UniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
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
                        [x["upostag"] for x in annotation],
                        [x["deps"][0] for x in annotation]
                        )

    @overrides
    def text_to_instance(self,  # type: ignore
                         words: List[str],
                         upos_tags: List[str],
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields["words"] = TextField([Token(w) for w in words], self._token_indexers)
        fields["pos_tags"] = SequenceLabelField(upos_tags, fields["words"], label_namespace="pos")
        fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                 fields["words"],
                                                 label_namespace="head_tags")
        fields["head_indices"] = SequenceLabelField([x[1] for x in dependencies],
                                                    fields["words"],
                                                    label_namespace="head_index_tags")
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'UniversalDependenciesDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return UniversalDependenciesDatasetReader(tokenizer=tokenizer,
                                                  token_indexers=token_indexers,
                                                  lazy=lazy)

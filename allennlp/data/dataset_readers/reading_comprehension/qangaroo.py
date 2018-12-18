import json
import logging

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField, MetadataField, IndexField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


@DatasetReader.register("qangaroo")
class QangarooReader(DatasetReader):
    """
    Reads a JSON-formatted Qangaroo file and returns a ``Dataset`` where the ``Instances`` have six
    fields: ``candidates``, a ``ListField[TextField]``, ``query``, a ``TextField``, ``supports``, a
    ``ListField[TextField]``, ``answer``, a ``TextField``, and ``answer_index``, a ``IndexField``.
    We also add a ``MetadataField`` that stores the instance's ID and annotations if they are present.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
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

        logger.info("Reading file at %s", file_path)

        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        logger.info("Reading the dataset")
        for sample in dataset:

            instance = self.text_to_instance(sample['candidates'], sample['query'], sample['supports'],
                                             sample['id'], sample['answer'],
                                             sample['annotations'] if 'annotations' in sample else [[]])

            yield instance

    @overrides
    def text_to_instance(self, # type: ignore
                         candidates: List[str],
                         query: str,
                         supports: List[str],
                         _id: str = None,
                         answer: str = None,
                         annotations: List[List[str]] = None) -> Instance:

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        candidates_field = ListField([TextField(candidate, self._token_indexers)
                                      for candidate in self._tokenizer.batch_tokenize(candidates)])

        fields['query'] = TextField(self._tokenizer.tokenize(query), self._token_indexers)

        fields['supports'] = ListField([TextField(support, self._token_indexers)
                                        for support in self._tokenizer.batch_tokenize(supports)])

        fields['answer'] = TextField(self._tokenizer.tokenize(answer), self._token_indexers)

        fields['answer_index'] = IndexField(candidates.index(answer), candidates_field)

        fields['candidates'] = candidates_field

        fields['metadata'] = MetadataField({'annotations': annotations, 'id': _id})

        return Instance(fields)

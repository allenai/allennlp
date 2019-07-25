from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import IndexField, LabelField, ListField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("next_token_lm")
class NextTokenLMReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # TODO(mattg): ACTUALLY IMPLEMENT THIS!! .
        with open(file_path, "r") as text_file:
            for sentence in text_file:
                tokens = self._tokenizer.tokenize(sentence)
                target = 'the'                
                yield self.text_to_instance(sentence, tokens, [target])

    @overrides
    def text_to_instance(self,
                         sentence: str = None,
                         tokens: List[Token] = None,
                         target: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        if not tokens:
            tokens = self._tokenizer.tokenize(sentence)                
        input_field = TextField(tokens, self._token_indexers)                        
        target_field = TextField([Token(target)], self._token_indexers)            
        return Instance({'tokens': input_field,
                         'target_ids': target_field})
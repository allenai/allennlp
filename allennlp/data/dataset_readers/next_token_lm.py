from typing import Dict, List
import logging

from overrides import overrides

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
class NextTokenLmReader(DatasetReader):
    """
    Creates ``Instances`` suitable for use in predicting a single next token using a language
    model.  The :class:`Field` s that we create are the following: an input ``TextField`` and a
    target token ``TextField`` (we only ver have a single token, but we use a ``TextField`` so we
    can index it the same way as our input, typically with a single
    ``PretrainedTransformerIndexer``).

    NOTE: This is not fully functional!  It was written to put together a demo for interpreting and
    attacking language models, not for actually training anything.  It would be a really bad idea
    to use this setup for training language models.  The only purpose of this class is for a demo.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text, and to get ids for the mask
        targets.  See :class:`TokenIndexer`.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # NOTE: this is only useful for easy model tests.  Implementing this function for real
        # doesn't make any sense, as you would never want to train a language model this way.
        with open(file_path, "r") as text_file:
            for sentence in text_file:
                tokens = self._tokenizer.tokenize(sentence)
                target = 'the'
                yield self.text_to_instance(sentence, tokens, target)

    @overrides
    def text_to_instance(self,
                         sentence: str = None,
                         tokens: List[Token] = None,
                         target: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        if not tokens:
            tokens = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokens, self._token_indexers)
        fields = {'tokens': input_field}
        if target:
            fields['target_ids'] = TextField([Token(target)], self._token_indexers)
        return Instance(fields)

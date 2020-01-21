from typing import Dict, List
import logging
import copy

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("next_token_lm")
class NextTokenLmReader(DatasetReader):
    """
    Creates `Instances` suitable for use in predicting a single next token using a language
    model.  The :class:`Field` s that we create are the following: an input `TextField` and a
    target token `TextField` (we only ver have a single token, but we use a `TextField` so we
    can index it the same way as our input, typically with a single
    `PretrainedTransformerIndexer`).

    NOTE: This is not fully functional!  It was written to put together a demo for interpreting and
    attacking language models, not for actually training anything.  It would be a really bad idea
    to use this setup for training language models, as it would be incredibly inefficient.  The
    only purpose of this class is for a demo.

    # Parameters

    tokenizer : `Tokenizer`, optional (default=`WhitespaceTokenizer()`)
        We use this `Tokenizer` for the text.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text, and to get ids for the mask
        targets.  See :class:`TokenIndexer`.
    """

    def __init__(
        self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._targets_tokenizer: Tokenizer
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            self._targets_tokenizer = copy.copy(self._tokenizer)
            self._targets_tokenizer._add_special_tokens = False
        else:
            self._targets_tokenizer = self._tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        import sys

        # You can call pytest with either `pytest` or `py.test`.
        if "test" not in sys.argv[0]:
            logger.error(
                "_read is only implemented for unit tests. You should not actually "
                "try to train or evaluate a language model with this code."
            )
        with open(file_path, "r") as text_file:
            for sentence in text_file:
                tokens = self._tokenizer.tokenize(sentence)
                target = "the"
                yield self.text_to_instance(sentence, tokens, target)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence: str = None,
        tokens: List[Token] = None,
        target: str = None,
    ) -> Instance:

        if not tokens:
            tokens = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {"tokens": input_field}
        # TODO: if we index word that was not split into wordpieces with
        # PretrainedTransformerTokenizer we will get OOV token ID...
        # Until this is handeled, let's use first wordpiece id for each token since tokens should contain text_ids
        # to be indexed with PretrainedTokenIndexer. It also requeires hack to avoid adding special tokens...
        if target:
            wordpiece = self._targets_tokenizer.tokenize(target)[0]
            target_token = Token(text=target, text_id=wordpiece.text_id, type_id=wordpiece.type_id)
            fields["target_ids"] = TextField([target_token], self._token_indexers)
        return Instance(fields)

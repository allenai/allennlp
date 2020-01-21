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
from allennlp.data.fields import IndexField, Field, ListField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("masked_language_modeling")
class MaskedLanguageModelingReader(DatasetReader):
    """
    Reads a text file and converts it into a `Dataset` suitable for training a masked language
    model.

    The :class:`Field` s that we create are the following: an input `TextField`, a mask position
    `ListField[IndexField]`, and a target token `TextField` (the target tokens aren't a single
    string of text, but we use a `TextField` so we can index the target tokens the same way as
    our input, typically with a single `PretrainedTransformerIndexer`).  The mask position and
    target token lists are the same length.

    NOTE: This is not fully functional!  It was written to put together a demo for interpreting and
    attacking masked language modeling, not for actually training anything.  `text_to_instance`
    is functional, but `_read` is not.  To make this fully functional, you would want some
    sampling strategies for picking the locations for [MASK] tokens, and probably a bunch of
    efficiency / multi-processing stuff.

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
        # temporary hack to not to add special tokens
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
            logger.error("_read is only implemented for unit tests at the moment")
        with open(file_path, "r") as text_file:
            for sentence in text_file:
                tokens = self._tokenizer.tokenize(sentence)
                target = tokens[0].text
                tokens[0] = Token("[MASK]")
                yield self.text_to_instance(sentence, tokens, [target])

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence: str = None,
        tokens: List[Token] = None,
        targets: List[str] = None,
    ) -> Instance:

        """
        # Parameters

        sentence : `str`, optional
            A sentence containing [MASK] tokens that should be filled in by the model.  This input
            is superceded and ignored if `tokens` is given.
        tokens : `List[Token]`, optional
            An already-tokenized sentence containing some number of [MASK] tokens to be predicted.
        targets : `List[str]`, optional
            Contains the target tokens to be predicted.  The length of this list should be the same
            as the number of [MASK] tokens in the input.
        """
        if not tokens:
            tokens = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokens, self._token_indexers)
        mask_positions = []
        for i, token in enumerate(tokens):
            if token.text == "[MASK]":
                mask_positions.append(i)
        if not mask_positions:
            raise ValueError("No [MASK] tokens found!")
        if targets and len(targets) != len(mask_positions):
            raise ValueError(f"Found {len(mask_positions)} mask tokens and {len(targets)} targets")
        mask_position_field = ListField([IndexField(i, input_field) for i in mask_positions])
        fields: Dict[str, Field] = {"tokens": input_field, "mask_positions": mask_position_field}
        # TODO(mattg): there's a problem if the targets get split into multiple word pieces...
        # (maksym-del): if we index word that was not split into wordpieces with
        # PretrainedTransformerTokenizer we will get OOV token ID...
        # Until this is handeled, let's use first wordpiece id for each token since tokens should contain text_ids
        # to be indexed with PretrainedTokenIndexer. It also requeires hack to avoid adding special tokens...
        if targets is not None:
            # target_field = TextField([Token(target) for target in targets], self._token_indexers)
            first_wordpieces = [self._targets_tokenizer.tokenize(target)[0] for target in targets]
            target_tokens = []
            for wordpiece, target in zip(first_wordpieces, targets):
                target_tokens.append(
                    Token(text=target, text_id=wordpiece.text_id, type_id=wordpiece.type_id)
                )
            fields["target_ids"] = TextField(target_tokens, self._token_indexers)
        return Instance(fields)

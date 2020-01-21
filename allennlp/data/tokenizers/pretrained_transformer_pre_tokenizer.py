from typing import List, Optional
from overrides import overrides

import spacy
from transformers.tokenization_bert import BasicTokenizer as BertTokenizer, _is_punctuation

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("bert-basic")
class BertPreTokenizer(Tokenizer):
    """
    The ``BasicTokenizer`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """

    default_never_split = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]

    def __init__(self, do_lower_case: bool = True, never_split: Optional[List[str]] = None) -> None:

        if never_split is None:
            never_split = self.default_never_split
        else:
            never_split = never_split + self.default_never_split

        self.basic_tokenizer = BertTokenizer(do_lower_case, never_split)
        self.basic_tokenizer._run_split_on_punc = self._run_split_on_punc
        self.never_split = never_split

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(text) for text in self.basic_tokenizer.tokenize(text)]

    # HACK: Monkeypatch for huggingface's broken BasicTokenizer.
    # TODO(Mark): Remove this once https://github.com/huggingface/transformers/pull/2557
    # is merged.
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is None:
            never_split = self.never_split
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]

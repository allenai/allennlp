from typing import List, Optional
from overrides import overrides

import spacy
import ftfy
from transformers.tokenization_bert import BasicTokenizer as BertTokenizer, _is_punctuation

from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize


@Tokenizer.register("openai")
class OpenAIPreTokenizer(Tokenizer):
    """
    For OpenAI transformer
    This is used to split a sentence into words.
    Then the ``OpenaiTransformerBytePairIndexer`` converts each word into wordpieces.
    """

    def __init__(self, language: str = "en_core_web_sm") -> None:
        self.spacy = get_spacy_model(language, False, False, False)

    @staticmethod
    def _standardize(text):
        return text_standardize(ftfy.fix_text(text))

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        standardized_sentences = [self._standardize(sentence) for sentence in texts]
        return [
            _remove_spaces(tokens)
            for tokens in self.spacy.pipe(standardized_sentences, n_threads=-1)
        ]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return _remove_spaces(self.spacy(self._standardize(text)))


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

from typing import List, Optional
from overrides import overrides

import spacy
import ftfy
from pytorch_pretrained_bert.tokenization import BasicTokenizer as BertTokenizer

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

    def __init__(self, do_lower_case: bool = True, never_split: Optional[List[str]] = None) -> None:
        if never_split is None:
            # Let BertTokenizer use its default
            self.basic_tokenizer = BertTokenizer(do_lower_case)
        else:
            self.basic_tokenizer = BertTokenizer(do_lower_case, never_split)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(text) for text in self.basic_tokenizer.tokenize(text)]


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]

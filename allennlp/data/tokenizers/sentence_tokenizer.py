from typing import List
from overrides import overrides
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


@Tokenizer.register("sentence")
class SentenceTokenizer(Tokenizer):
    """
    A ``SentenceTokenizer`` handles the splitting of strings into sentences.

    Parameters
    ----------
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._start_tokens = start_tokens or []
        self._end_tokens = end_tokens or []
        self._sentence_splitter = SpacySentenceSplitter()

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This uses a ``SentenceSplitter`` to split text into sentences.
        """
        sents = self._sentence_splitter.split_sentences(text)
        for start_token in self._start_tokens:
            sents.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            sents.append(Token(end_token, -1))
        return sents

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        batched_sents = self._sentence_splitter.batch_split_sentences(texts)
        for sents in batched_sents:
            for sent in sents:
                for start_token in self._start_tokens:
                    sent.insert(0, Token(start_token, 0))
                for end_token in self._end_tokens:
                    sent.append(Token(end_token, -1))
        return batched_sents

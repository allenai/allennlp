from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter, SpacyWordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer


@Tokenizer.register("word")
class WordTokenizer(Tokenizer):
    """
    A ``WordTokenizer`` handles the splitting of strings into words as well as any desired
    post-processing (e.g., stemming, filtering, etc.).  Note that we leave one particular piece of
    post-processing for later: the decision of whether or not to lowercase the token.  This is for
    two reasons: (1) if you want to make two different casing decisions for whatever reason, you
    won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
    for your word embedding, but retain capitalization in a character-level representation, we need
    to retain the capitalization here.

    Parameters
    ----------
    word_splitter : ``WordSplitter``, optional
        The :class:`WordSplitter` to use for splitting text strings into word tokens.  The default
        is to use the ``SpacyWordSplitter`` with default parameters.
    word_filter : ``WordFilter``, optional
        The :class:`WordFilter` to use for, e.g., removing stopwords.  Default is to do no
        filtering.
    word_stemmer : ``WordStemmer``, optional
        The :class:`WordStemmer` to use.  Default is no stemming.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self,
                 word_splitter: WordSplitter = None,
                 word_filter: WordFilter = PassThroughWordFilter(),
                 word_stemmer: WordStemmer = PassThroughWordStemmer(),
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._word_splitter = word_splitter or SpacyWordSplitter()
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Does whatever processing is required to convert a string of text into a sequence of tokens.

        At a minimum, this uses a ``WordSplitter`` to split words into text.  It may also do
        stemming or stopword removal, depending on the parameters given to the constructor.
        """
        words = self._word_splitter.split_words(text)
        return self._filter_and_stem(words)

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        batched_words = self._word_splitter.batch_split_words(texts)
        return [self._filter_and_stem(words) for words in batched_words]

    def _filter_and_stem(self, words: List[Token]) -> List[Token]:
        filtered_words = self._word_filter.filter_words(words)
        stemmed_words = [self._word_stemmer.stem_word(word) for word in filtered_words]
        for start_token in self._start_tokens:
            stemmed_words.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            stemmed_words.append(Token(end_token, -1))
        return stemmed_words

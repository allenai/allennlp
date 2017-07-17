from typing import List

from allennlp.common import Params
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer


class WordTokenizer(Tokenizer):
    """
    A ``WordTokenizer`` handles the splitting of strings into words (with the use of a
    WordSplitter) as well as any desired post-processing (e.g., stemming, filtering, etc.).  Note
    that we leave one particular piece of post-processing for later: the decision of whether or not
    to lowercase the token.  This is for two reasons: (1) if you want to make two different casing
    decisions for whatever reason, you won't have to run the tokenizer twice, and more importantly
    (2) if you want to lowercase words for your word embedding, but retain capitalization in a
    character-level representation, we need to retain the capitalization here.

    Parameters
    ----------
    word_splitter : ``WordSplitter``, optional (default=``SimpleWordSplitter``)
        The :class:`WordSplitter` to use for splitting text strings into word tokens.

    word_filter : ``WordFilter``, optional (default=``PassThroughWordFilter``)
        The :class:`WordFilter` to use for, e.g., removing stopwords.  Default is to do no
        filtering.

    word_stemmer : ``WordStemmer``, optional (default=``PassThroughWordStemmer``)
        The :class:`WordStemmer` to use.  Default is no stemming.
    """
    def __init__(self,
                 word_splitter: WordSplitter = SimpleWordSplitter(),
                 word_filter: WordFilter = PassThroughWordFilter(),
                 word_stemmer: WordStemmer = PassThroughWordStemmer()) -> None:
        self.word_splitter = word_splitter
        self.word_filter = word_filter
        self.word_stemmer = word_stemmer

    def tokenize(self, text: str) -> List[str]:
        """
        Does whatever processing is required to convert a string of text into a sequence of tokens.

        At a minimum, this uses a ``WordSplitter`` to split words into text.  It may also do
        stemming or stopword removal, depending on the parameters given to the constructor.
        """
        words = self.word_splitter.split_words(text)
        filtered_words = self.word_filter.filter_words(words)
        stemmed_words = [self.word_stemmer.stem_word(word) for word in filtered_words]
        return stemmed_words

    @classmethod
    def from_params(cls, params: Params) -> 'WordTokenizer':
        """
        Parameters
        ----------
        word_splitter : ``str``, default=``"simple"``
            The string name of the ``WordSplitter`` of choice (see the options at the bottom of
            ``word_splitter.py``).

        word_filter : ``str``, default=``"pass_through"``
            The name of the ``WordFilter`` to use (see the options at the bottom of
            ``word_filter.py``).

        word_stemmer : ``str``, default=``"pass_through"``
            The name of the ``WordStemmer`` to use (see the options at the bottom of
            ``word_stemmer.py``).
        """
        word_splitter = WordSplitter.from_params(params.pop('word_splitter', {}))
        word_filter = WordFilter.from_params(params.pop('word_filter', {}))
        word_stemmer = WordStemmer.from_params(params.pop('word_stemmer', {}))
        params.assert_empty(cls.__name__)
        return cls(word_splitter=word_splitter, word_filter=word_filter, word_stemmer=word_stemmer)

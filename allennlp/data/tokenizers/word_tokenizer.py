from typing import List, Tuple

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter, SpacyWordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer


@Tokenizer.register("word")
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
    word_splitter : ``WordSplitter``, optional
        The :class:`WordSplitter` to use for splitting text strings into word tokens.  The default
        is to use the spacy word splitter.
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
                 word_splitter: WordSplitter = SpacyWordSplitter(),
                 word_filter: WordFilter = PassThroughWordFilter(),
                 word_stemmer: WordStemmer = PassThroughWordStemmer(),
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._word_splitter = word_splitter
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Does whatever processing is required to convert a string of text into a sequence of tokens.

        At a minimum, this uses a ``WordSplitter`` to split words into text.  It may also do
        stemming or stopword removal, depending on the parameters given to the constructor.
        """
        words, offsets = self._word_splitter.split_words(text)
        words_to_keep = self._word_filter.should_keep_words(words)
        filtered_words = []
        filtered_offsets = []
        for i, should_keep in enumerate(words_to_keep):
            if should_keep:
                filtered_words.append(words[i])
                filtered_offsets.append(offsets[i])
        stemmed_words = [self._word_stemmer.stem_word(word) for word in filtered_words]
        for start_token in self._start_tokens:
            stemmed_words.insert(0, start_token)
            if filtered_offsets is not None:
                filtered_offsets.insert(0, (0, 0))
        for end_token in self._end_tokens:
            stemmed_words.append(end_token)
            if filtered_offsets is not None:
                filtered_offsets.append((-1, -1))
        return stemmed_words, filtered_offsets

    @classmethod
    def from_params(cls, params: Params) -> 'WordTokenizer':
        word_splitter = WordSplitter.from_params(params.pop('word_splitter', {}))
        word_filter = WordFilter.from_params(params.pop('word_filter', {}))
        word_stemmer = WordStemmer.from_params(params.pop('word_stemmer', {}))
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        params.assert_empty(cls.__name__)
        return cls(word_splitter=word_splitter,
                   word_filter=word_filter,
                   word_stemmer=word_stemmer,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens)

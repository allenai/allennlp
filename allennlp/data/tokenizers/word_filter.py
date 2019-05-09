from typing import List
import re
from overrides import overrides

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.common.file_utils import read_set_from_file


class WordFilter(Registrable):
    """
    A ``WordFilter`` removes words from a token list.  Typically, this is for stopword removal,
    though you could feasibly use it for more domain-specific removal if you want.

    Word removal happens `before` stemming, so keep that in mind if you're designing a list of
    words to be removed.
    """
    default_implementation = 'pass_through'

    def filter_words(self, words: List[Token]) -> List[Token]:
        """
        Returns a filtered list of words.
        """
        raise NotImplementedError


@WordFilter.register('pass_through')
class PassThroughWordFilter(WordFilter):
    """
    Does not filter words; it's a no-op.  This is the default word filter.
    """
    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return words


@WordFilter.register('regex')
class RegexFilter(WordFilter):
    """
    A ``RegexFilter`` removes words according to supplied regex patterns.

    Parameters
    ----------
    patterns : ``List[str]``
        Words matching these regex patterns will be removed as stopwords.
    """
    def __init__(self,
                 patterns: List[str]) -> None:
        self._patterns = patterns
        self._joined_pattern = re.compile("|".join(self._patterns))

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        stopwords = [word for word in words
                     if not self._joined_pattern.match(word.text)]
        return stopwords


@WordFilter.register('stopwords')
class StopwordFilter(WordFilter):
    """
    A ``StopwordFilter`` uses a list of stopwords to filter.
    If no file is specified, nltk's default list of stopwords is used.
    Words and stopwords are lowercased for comparison.

    Parameters
    ----------
    stopword_file : ``str``, optional
        A filename containing stopwords to filter out (file format is one stopword per line).
    tokens_to_add : ``List[str]``, optional
        A list of tokens to additionally filter out.
    """
    def __init__(self,
                 stopword_file: str = None,
                 tokens_to_add: List[str] = None) -> None:
        self._tokens_to_add = tokens_to_add or []
        if stopword_file is not None:
            self.stopwords = {token.lower() for token in read_set_from_file(stopword_file)}
        else:
            self.stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                              "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
                              'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                              'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                              'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                              'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                              'about', 'against', 'between', 'into', 'through', 'during', 'before',
                              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                              'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                              'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                              'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                              'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                              'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                              'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                              'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                              "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                              "weren't", 'won', "won't", 'wouldn', "wouldn't"}
        for token in self._tokens_to_add:
            self.stopwords.add(token.lower())

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return [word for word in words if word.text.lower() not in self.stopwords]

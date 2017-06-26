from collections import OrderedDict

from nltk.stem import PorterStemmer as NltkPorterStemmer
from overrides import overrides

from ...common import Params


class WordStemmer:
    """
    A ``WordStemmer`` lemmatizes words.  This means that we map words to their root form, so that,
    e.g., "have", "has", and "had" all have the same internal representation.

    You should think carefully about whether and how much stemming you want in your model.  Kind of
    the whole point of using word embeddings is so that you don't have to do this, but in a highly
    inflected language, or in a low-data setting, you might need it anyway.  The default
    ``WordStemmer`` does nothing, just returning the work token as-is.
    """
    def stem_word(self, word: str) -> str:
        """Converts a word to its lemma"""
        raise NotImplementedError

    @staticmethod
    def from_params(params: Params) -> 'WordStemmer':
        choice = params.pop_choice('type', list(word_stemmers.keys()), default_to_first_choice=True)
        params.assert_empty('WordStemmer')
        return word_stemmers[choice]()


class PassThroughWordStemmer(WordStemmer):
    """
    Does not stem words; it's a no-op.  This is the default word stemmer.
    """
    @overrides
    def stem_word(self, word: str) -> str:
        return word


class PorterStemmer(WordStemmer):
    """
    Uses NLTK's PorterStemmer to stem words.
    """
    def __init__(self):
        self.stemmer = NltkPorterStemmer()

    @overrides
    def stem_word(self, word: str) -> str:
        return self.stemmer.stem(word)


word_stemmers = OrderedDict()  # pylint: disable=invalid-name
word_stemmers['pass_through'] = PassThroughWordStemmer
word_stemmers['porter'] = PorterStemmer

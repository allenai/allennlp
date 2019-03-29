from nltk.stem import PorterStemmer as NltkPorterStemmer
from overrides import overrides

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


class WordStemmer(Registrable):
    """
    A ``WordStemmer`` lemmatizes words.  This means that we map words to their root form, so that,
    e.g., "have", "has", and "had" all have the same internal representation.

    You should think carefully about whether and how much stemming you want in your model.  Kind of
    the whole point of using word embeddings is so that you don't have to do this, but in a highly
    inflected language, or in a low-data setting, you might need it anyway.  The default
    ``WordStemmer`` does nothing, just returning the work token as-is.
    """
    default_implementation = 'pass_through'

    def stem_word(self, word: Token) -> Token:
        """
        Returns a new ``Token`` with ``word.text`` replaced by a stemmed word.
        """
        raise NotImplementedError


@WordStemmer.register('pass_through')
class PassThroughWordStemmer(WordStemmer):
    """
    Does not stem words; it's a no-op.  This is the default word stemmer.
    """
    @overrides
    def stem_word(self, word: Token) -> Token:
        return word


@WordStemmer.register('porter')
class PorterStemmer(WordStemmer):
    """
    Uses NLTK's PorterStemmer to stem words.
    """
    def __init__(self):
        self.stemmer = NltkPorterStemmer()

    @overrides
    def stem_word(self, word: Token) -> Token:
        new_text = self.stemmer.stem(word.text)
        return Token(text=new_text,
                     idx=word.idx,
                     lemma_=word.lemma_,
                     pos_=word.pos_,
                     tag_=word.tag_,
                     dep_=word.dep_,
                     ent_type_=word.ent_type_,
                     text_id=getattr(word, 'text_id', None))

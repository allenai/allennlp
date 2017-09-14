from typing import List

from overrides import overrides

from allennlp.common import Params, Registrable
from allennlp.data.tokenizers.token import Token


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

    @classmethod
    def from_params(cls, params: Params) -> 'WordFilter':
        choice = params.pop_choice('type', cls.list_available(), default_to_first_choice=True)
        params.assert_empty('WordFilter')
        return cls.by_name(choice)()


@WordFilter.register('pass_through')
class PassThroughWordFilter(WordFilter):
    """
    Does not filter words; it's a no-op.  This is the default word filter.
    """
    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return words


@WordFilter.register('stopwords')
class StopwordFilter(WordFilter):
    """
    Uses a list of stopwords to filter.
    """
    def __init__(self):
        # TODO(matt): Allow this to be specified somehow, either with a file, or with parameters,
        # or something.
        self.stopwords = set(['I', 'a', 'aboard', 'about', 'above', 'accordance', 'according',
                              'across', 'after', 'against', 'along', 'alongside', 'also', 'am',
                              'amid', 'amidst', 'an', 'and', 'apart', 'are', 'around', 'as',
                              'aside', 'astride', 'at', 'atop', 'back', 'be', 'because', 'before',
                              'behind', 'below', 'beneath', 'beside', 'besides', 'between',
                              'beyond', 'but', 'by', 'concerning', 'do', 'down', 'due', 'during',
                              'either', 'except', 'exclusive', 'false', 'for', 'from', 'happen',
                              'he', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                              'how many', 'how much', 'i', 'if', 'in', 'including', 'inside',
                              'instead', 'into', 'irrespective', 'is', 'it', 'its', 'itself',
                              'less', 'me', 'mine', 'minus', 'my', 'myself', 'neither', 'next',
                              'not', 'occur', 'of', 'off', 'on', 'onto', 'opposite', 'or', 'our',
                              'ours', 'ourselves', 'out', 'out of', 'outside', 'over', 'owing',
                              'per', 'prepatory', 'previous', 'prior', 'pursuant', 'regarding',
                              's', 'sans', 'she', 'subsequent', 'such', 'than', 'thanks', 'that',
                              'the', 'their', 'theirs', 'them', 'themselves', 'then', 'these',
                              'they', 'this', 'those', 'through', 'throughout', 'thru', 'till',
                              'to', 'together', 'top', 'toward', 'towards', 'true', 'under',
                              'underneath', 'unlike', 'until', 'up', 'upon', 'us', 'using',
                              'versus', 'via', 'was', 'we', 'were', 'what', 'when', 'where',
                              'which', 'who', 'why', 'will', 'with', 'within', 'without', 'you',
                              'your', 'yours', 'yourself', 'yourselves', ",", '.', ':', '!', ';',
                              "'", '"', '&', '$', '#', '@', '(', ')', '?'])

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return [word for word in words if word.text.lower() not in self.stopwords]

from collections import OrderedDict
from typing import List

from overrides import overrides

from ...common import Params

class WordSplitter:
    """
    A ``WordSplitter`` splits strings into words.  This is typically called a "tokenizer" in NLP,
    because splitting strings into characters is trivial, but we use ``Tokenizer`` to refer to the
    higher-level object that splits strings into tokens (which could just be character tokens).
    So, we're using "word splitter" here for this.
    """
    def split_words(self, sentence: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def from_params(params: Params) -> 'WordSplitter':
        choice = params.pop_choice('type', list(word_splitters.keys()), default_to_first_choice=True)
        params.assert_empty('WordSplitter')
        return word_splitters[choice]()


class SimpleWordSplitter(WordSplitter):
    """
    Does really simple tokenization.  NLTK was too slow, so we wrote our own simple tokenizer
    instead.  This just does an initial split(), followed by some heuristic filtering of each
    whitespace-delimited token, separating contractions and punctuation.  We assume lower-cased,
    reasonably well-formed English sentences as input.
    """
    def __init__(self):
        # These are certainly incomplete.  But at least it's a start.
        self.special_cases = set(['mr.', 'mrs.', 'etc.', 'e.g.', 'cf.', 'c.f.', 'eg.', 'al.'])
        self.contractions = set(["n't", "'s", "'ve", "'re", "'ll", "'d", "'m"])
        self.contractions |= set([x.replace("'", "’") for x in self.contractions])
        self.ending_punctuation = set(['"', "'", '.', ',', ';', ')', ']', '}', ':', '!', '?', '%', '”', "’"])
        self.beginning_punctuation = set(['"', "'", '(', '[', '{', '#', '$', '“', "‘"])

    @overrides
    def split_words(self, sentence: str) -> List[str]:
        """
        Splits a sentence into word tokens.  We handle four kinds of things: words with punctuation
        that should be ignored as a special case (Mr. Mrs., etc.), contractions/genitives (isn't,
        don't, Matt's), and beginning and ending punctuation ("antennagate", (parentheticals), and
        such.).

        The basic outline is to split on whitespace, then check each of these cases.  First, we
        strip off beginning punctuation, then strip off ending punctuation, then strip off
        contractions.  When we strip something off the beginning of a word, we can add it to the
        list of tokens immediately.  When we strip it off the end, we have to save it to be added
        to after the word itself has been added.  Before stripping off any part of a token, we
        first check to be sure the token isn't in our list of special cases.
        """
        fields = sentence.lower().split()
        tokens = []
        for field in fields:  # type: str
            add_at_end = []
            while self._can_split(field) and field[0] in self.beginning_punctuation:
                tokens.append(field[0])
                field = field[1:]
            while self._can_split(field) and field[-1] in self.ending_punctuation:
                add_at_end.insert(0, field[-1])
                field = field[:-1]

            # There could (rarely) be several contractions in a word, but we check contractions
            # sequentially, in a random order.  If we've removed one, we need to check again to be
            # sure there aren't others.
            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in self.contractions:
                    if self._can_split(field) and field.endswith(contraction):
                        field = field[:-len(contraction)]
                        add_at_end.insert(0, contraction)
                        remove_contractions = True
            if field:
                tokens.append(field)
            tokens.extend(add_at_end)
        return tokens

    def _can_split(self, token: str):
        return token and token not in self.special_cases


class SpaceWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.

    Note that we use ``sentence.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """
    @overrides
    def split_words(self, sentence: str) -> List[str]:
        return sentence.split()


class NltkWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses nltk's ``word_tokenize`` method.

    I found that nltk is very slow, so I switched to using my own simple one, which is a good deal
    faster.  But I'm adding this one back so that there's consistency with older versions of the
    code, if you really want it.
    """
    @overrides
    def split_words(self, sentence: str) -> List[str]:
        # Import is here because it's slow, and by default unnecessary.
        from nltk.tokenize import word_tokenize
        return word_tokenize(sentence.lower())


class SpacyWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses spaCy's Tokenizer, which is much faster than the others.
    """
    def __init__(self):
        # Import is here it's slow, and can be unnecessary.
        import spacy
        self.en_nlp = spacy.load('en')

    @overrides
    def split_words(self, sentence: str) -> List[str]:
        return [str(token.lower_) for token in self.en_nlp.tokenizer(sentence)]


word_splitters = OrderedDict()  # pylint: disable=invalid-name
word_splitters['simple'] = SimpleWordSplitter
word_splitters['spaces'] = SpaceWordSplitter
word_splitters['nltk'] = NltkWordSplitter
word_splitters['spacy'] = SpacyWordSplitter

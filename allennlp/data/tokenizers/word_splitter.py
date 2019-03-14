import re
from typing import List

from overrides import overrides
import spacy
import ftfy

from pytorch_pretrained_bert.tokenization import BasicTokenizer as BertTokenizer

from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize


class WordSplitter(Registrable):
    """
    A ``WordSplitter`` splits strings into words.  This is typically called a "tokenizer" in NLP,
    because splitting strings into characters is trivial, but we use ``Tokenizer`` to refer to the
    higher-level object that splits strings into tokens (which could just be character tokens).
    So, we're using "word splitter" here for this.
    """
    default_implementation = 'spacy'

    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        """
        Spacy needs to do batch processing, or it can be really slow.  This method lets you take
        advantage of that if you want.  Default implementation is to just iterate of the sentences
        and call ``split_words``, but the ``SpacyWordSplitter`` will actually do batched
        processing.
        """
        return [self.split_words(sentence) for sentence in sentences]

    def split_words(self, sentence: str) -> List[Token]:
        """
        Splits ``sentence`` into a list of :class:`Token` objects.
        """
        raise NotImplementedError


@WordSplitter.register('simple')
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
    def split_words(self, sentence: str) -> List[Token]:
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
        fields = sentence.split()
        tokens: List[Token] = []
        for field in fields:
            add_at_end: List[Token] = []
            while self._can_split(field) and field[0] in self.beginning_punctuation:
                tokens.append(Token(field[0]))
                field = field[1:]
            while self._can_split(field) and field[-1] in self.ending_punctuation:
                add_at_end.insert(0, Token(field[-1]))
                field = field[:-1]

            # There could (rarely) be several contractions in a word, but we check contractions
            # sequentially, in a random order.  If we've removed one, we need to check again to be
            # sure there aren't others.
            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in self.contractions:
                    if self._can_split(field) and field.lower().endswith(contraction):
                        add_at_end.insert(0, Token(field[-len(contraction):]))
                        field = field[:-len(contraction)]
                        remove_contractions = True
            if field:
                tokens.append(Token(field))
            tokens.extend(add_at_end)
        return tokens

    def _can_split(self, token: str):
        return token and token.lower() not in self.special_cases


@WordSplitter.register('letters_digits')
class LettersDigitsWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` which keeps runs of (unicode) letters and runs of digits together, while
    every other non-whitespace character becomes a separate word.
    """
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # We use the [^\W\d_] pattern as a trick to match unicode letters
        tokens = [Token(m.group(), idx=m.start())
                  for m in re.finditer(r'[^\W\d_]+|\d+|\S', sentence)]
        return tokens


@WordSplitter.register('just_spaces')
class JustSpacesWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.  We use a somewhat odd name here to avoid coming too close to the more
    commonly used ``SpacyWordSplitter``.

    Note that we use ``sentence.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(t) for t in sentence.split()]


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]


@WordSplitter.register('spacy')
class SpacyWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended ``WordSplitter``. By default it will return allennlp Tokens,
    which are small, efficient NamedTuples (and are serializable). If you want
    to keep the original spaCy tokens, pass keep_spacy_tokens=True.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False,
                 keep_spacy_tokens: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
        self._keep_spacy_tokens = keep_spacy_tokens

    def _sanitize(self, tokens: List[spacy.tokens.Token]) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        if self._keep_spacy_tokens:
            return tokens
        else:
            return [Token(token.text,
                          token.idx,
                          token.lemma_,
                          token.pos_,
                          token.tag_,
                          token.dep_,
                          token.ent_type_) for token in tokens]

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [self._sanitize(_remove_spaces(tokens))
                for tokens in self.spacy.pipe(sentences, n_threads=-1)]

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return self._sanitize(_remove_spaces(self.spacy(sentence)))


@WordSplitter.register('openai')
class OpenAISplitter(WordSplitter):
    """
    For OpenAI transformer
    """
    def __init__(self, language: str = 'en_core_web_sm') -> None:
        self.spacy = get_spacy_model(language, False, False, False)

    @staticmethod
    def _standardize(text):
        return text_standardize(ftfy.fix_text(text))

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        standardized_sentences = [self._standardize(sentence) for sentence in sentences]
        return [_remove_spaces(tokens)
                for tokens in self.spacy.pipe(standardized_sentences, n_threads=-1)]

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return _remove_spaces(self.spacy(self._standardize(sentence)))


@WordSplitter.register("bert-basic")
class BertBasicWordSplitter(WordSplitter):
    """
    The ``BasicWordSplitter`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """
    def __init__(self, do_lower_case: bool = True) -> None:
        self.basic_tokenizer = BertTokenizer(do_lower_case)

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(text) for text in self.basic_tokenizer.tokenize(sentence)]

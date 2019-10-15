import re
from typing import List, Optional

from overrides import overrides
import spacy
from spacy.tokens import Doc
import ftfy

from pytorch_pretrained_bert.tokenization import BasicTokenizer as BertTokenizer

from allennlp.common import Registrable, Params
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize

@Tokenizer.register("word")
class SpacyWordTokenizer(Tokenizer):
    """
    A ``Tokenizer`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended ``Tokenizer``. By default it will return allennlp Tokens,
    which are small, efficient NamedTuples (and are serializable). If you want
    to keep the original spaCy tokens, pass keep_spacy_tokens=True.
    """

    def __init__(
        self,
        language: str = "en_core_web_sm",
        pos_tags: bool = False,
        parse: bool = False,
        ner: bool = False,
        keep_spacy_tokens: bool = False,
        split_on_spaces: bool = False,
    ) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
        if split_on_spaces:
            self.spacy.tokenizer = WhitespaceSpacyTokenizer(self.spacy.vocab)

        self._keep_spacy_tokens = keep_spacy_tokens

    def _sanitize(self, tokens: List[spacy.tokens.Token]) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        if self._keep_spacy_tokens:
            return tokens
        else:
            return [
                Token(
                    token.text,
                    token.idx,
                    token.lemma_,
                    token.pos_,
                    token.tag_,
                    token.dep_,
                    token.ent_type_,
                )
                for token in tokens
            ]

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [
            self._sanitize(_remove_spaces(tokens))
            for tokens in self.spacy.pipe(texts, n_threads=-1)
        ]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return self._sanitize(_remove_spaces(self.spacy(text)))
    
    @classmethod
    def from_params(cls, params: Params) -> "SpacyWordTokenizer":  # type: ignore

        # Backwards compatibility for legacy "word" Tokenizer
        # which provided constructor arguments inside "word_splitter" key.
        word_splitter_params = params.get('word_splitter')
        if word_splitter_params:
            params = word_splitter_params
        
        language = params.pop("language", "en_core_web_sm")
        pos_tags = params.pop_bool("pos_tags", False)
        parse = params.pop_bool("parse", False)
        ner = params.pop_bool("ner", False)
        keep_spacy_tokens = params.pop_bool("keep_spacy_tokens", False)
        split_on_spaces = params.pop_bool("split_on_spaces", False)
        
        params.assert_empty(cls.__name__)

        return cls(
            language=language,
            pos_tags=pos_tags,
            parse=parse,
            ner=ner,
            keep_spacy_tokens=keep_spacy_tokens,
            split_on_spaces=split_on_spaces
        )


@Tokenizer.register("simple")
class SimpleWordTokenizer(Tokenizer):
    """
    Does really simple tokenization.  NLTK was too slow, so we wrote our own simple tokenizer
    instead.  This just does an initial split(), followed by some heuristic filtering of each
    whitespace-delimited token, separating contractions and punctuation.  We assume lower-cased,
    reasonably well-formed English sentences as input.
    """

    def __init__(self):
        # These are certainly incomplete.  But at least it's a start.
        self.special_cases = {"mr.", "mrs.", "etc.", "e.g.", "cf.", "c.f.", "eg.", "al."}
        self.contractions = {"n't", "'s", "'ve", "'re", "'ll", "'d", "'m"}
        self.contractions |= {x.replace("'", "’") for x in self.contractions}
        self.ending_punctuation = {
            '"',
            "'",
            ".",
            ",",
            ";",
            ")",
            "]",
            "}",
            ":",
            "!",
            "?",
            "%",
            "”",
            "’",
        }
        self.beginning_punctuation = {'"', "'", "(", "[", "{", "#", "$", "“", "‘"}

    @overrides
    def tokenize(self, text: str) -> List[Token]:
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
        fields = text.split()
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
                        add_at_end.insert(0, Token(field[-len(contraction) :]))
                        field = field[: -len(contraction)]
                        remove_contractions = True
            if field:
                tokens.append(Token(field))
            tokens.extend(add_at_end)
        return tokens

    def _can_split(self, token: str):
        return token and token.lower() not in self.special_cases


@Tokenizer.register("letters_digits")
class LettersDigitsTokenizer(Tokenizer):
    """
    A ``Tokenizer`` which keeps runs of (unicode) letters and runs of digits together, while
    every other non-whitespace character becomes a separate word.
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # We use the [^\W\d_] pattern as a trick to match unicode letters
        tokens = [
            Token(m.group(), idx=m.start()) for m in re.finditer(r"[^\W\d_]+|\d+|\S", sentence)
        ]
        return tokens


@Tokenizer.register("just_spaces")
class JustSpacesTokenizer(Tokenizer):
    """
    A ``Tokenizer`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.  We use a somewhat odd name here to avoid coming too close to the more
    commonly used ``SpacyTokenizer``.

    Note that we use ``text.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in text.split()]


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]


class WhitespaceSpacyTokenizer:
    """
    Spacy doesn't assume that text is tokenised. Sometimes this
    is annoying, like when you have gold data which is pre-tokenised,
    but Spacy's tokenisation doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = WhitespaceSpacyTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

@Tokenizer.register("openai")
class OpenAITokenizer(Tokenizer):
    """
    For OpenAI transformer
    """

    def __init__(self, language: str = "en_core_web_sm") -> None:
        self.spacy = get_spacy_model(language, False, False, False)

    @staticmethod
    def _standardize(text):
        return text_standardize(ftfy.fix_text(text))

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        standardized_sentences = [self._standardize(sentence) for sentence in texts]
        return [
            _remove_spaces(tokens)
            for tokens in self.spacy.pipe(standardized_sentences, n_threads=-1)
        ]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return _remove_spaces(self.spacy(self._standardize(sentence)))


@Tokenizer.register("bert-basic")
class BertBasicTokenizer(Tokenizer):
    """
    The ``BasicTokenizer`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """

    def __init__(self, do_lower_case: bool = True, never_split: Optional[List[str]] = None) -> None:
        if never_split is None:
            # Let BertTokenizer use its default
            self.basic_tokenizer = BertTokenizer(do_lower_case)
        else:
            self.basic_tokenizer = BertTokenizer(do_lower_case, never_split)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(text) for text in self.basic_tokenizer.tokenize(text)]



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
    word_splitter : ``Tokenizer``, optional
        The :class:`Tokenizer` to use for splitting text strings into word tokens.  The default
        is to use the ``SpacyTokenizer`` with default parameters.
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

    def __init__(
        self,
        word_splitter: Tokenizer = None,
        word_filter: WordFilter = PassThroughWordFilter(),
        word_stemmer: WordStemmer = PassThroughWordStemmer(),
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
    ) -> None:
        self._word_splitter = word_splitter or SpacyTokenizer()
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

        At a minimum, this uses a ``Tokenizer`` to split words into text.  It may also do
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

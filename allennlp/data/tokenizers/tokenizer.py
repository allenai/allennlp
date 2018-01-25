from typing import List

from allennlp.common import Params, Registrable
from allennlp.data.tokenizers.token import Token


class Tokenizer(Registrable):
    """
    A ``Tokenizer`` splits strings of text into tokens.  Typically, this either splits text into
    word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
    here, though you could imagine wanting to do other kinds of tokenization for structured or
    other inputs.

    As part of tokenization, concrete implementations of this API will also handle stemming,
    stopword filtering, adding start and end tokens, or other kinds of things you might want to do
    to your tokens.  See the parameters to, e.g., :class:`~.WordTokenizer`, or whichever tokenizer
    you want to use.

    If the base input to your model is words, you should use a :class:`~.WordTokenizer`, even if
    you also want to have a character-level encoder to get an additional vector for each word
    token.  Splitting word tokens into character arrays is handled separately, in the
    :class:`..token_representations.TokenRepresentation` class.
    """
    default_implementation = 'word'

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        """
        Batches together tokenization of several texts, in case that is faster for particular
        tokenizers.
        """
        raise NotImplementedError

    def tokenize(self, text: str) -> List[Token]:
        """
        Actually implements splitting words into tokens.

        Returns
        -------
        tokens : ``List[Token]``
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'Tokenizer':
        choice = params.pop_choice('type', cls.list_available(), default_to_first_choice=True)
        return cls.by_name(choice).from_params(params)

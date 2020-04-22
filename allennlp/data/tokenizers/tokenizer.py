from typing import List
import logging

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


logger = logging.getLogger(__name__)


class Tokenizer(Registrable):
    """
    A `Tokenizer` splits strings of text into tokens.  Typically, this either splits text into
    word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
    here, though you could imagine wanting to do other kinds of tokenization for structured or
    other inputs.

    See the parameters to, e.g., :class:`~.SpacyTokenizer`, or whichever tokenizer
    you want to use.

    If the base input to your model is words, you should use a :class:`~.SpacyTokenizer`, even if
    you also want to have a character-level encoder to get an additional vector for each word
    token.  Splitting word tokens into character arrays is handled separately, in the
    :class:`..token_representations.TokenRepresentation` class.
    """

    default_implementation = "spacy"

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        """
        Batches together tokenization of several texts, in case that is faster for particular
        tokenizers.

        By default we just do this without batching.  Override this in your tokenizer if you have a
        good way of doing batched computation.
        """
        return [self.tokenize(text) for text in texts]

    def tokenize(self, text: str) -> List[Token]:
        """
        Actually implements splitting words into tokens.

        # Returns

        tokens : `List[Token]`
        """
        raise NotImplementedError

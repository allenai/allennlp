from typing import List, Optional
import logging

from allennlp.common import Registrable
from allennlp.data.tokenizers.token_class import Token


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

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        """
        Adds special tokens to tokenized text. These are tokens like [CLS] or [SEP].

        Not all tokenizers do this. The default is to just return the tokens unchanged.

        # Parameters

        tokens1 : `List[Token]`
            The list of tokens to add special tokens to.
        tokens2 : `Optional[List[Token]]`
            An optional second list of tokens. This will be concatenated with `tokens1`. Special tokens will be
            added as appropriate.

        # Returns
        tokens : `List[Token]`
            The combined list of tokens, with special tokens added.
        """
        return tokens1 + (tokens2 or [])

    def num_special_tokens_for_sequence(self) -> int:
        """
        Returns the number of special tokens added for a single sequence.
        """
        return 0

    def num_special_tokens_for_pair(self) -> int:
        """
        Returns the number of special tokens added for a pair of sequences.
        """
        return 0

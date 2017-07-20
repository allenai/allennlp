import torch

from allennlp.common import Params
from allennlp.data import Vocabulary

class TokenVectorizer(torch.nn.Module):
    """
    A ``TokenVectorizer`` is a ``Module`` that takes as input a tensor with integer ids that have
    been output from a :class:`~allennutlp.data.TokenIndexer` and outputs a vector per token in the
    input.  The input typically has shape ``(batch_size, num_tokens)`` or ``(batch_size,
    num_tokens, num_characters)``, and the output is of shape ``(batch_size, num_tokens,
    output_dim)``.  The simplest ``TokenVectorizer`` is just an embedding layer, but for
    character-level input, it could also be some kind of character encoder.

    We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  This lets us
    more easily compute output dimensions for the :class:`~allennlp.modules.TokenEmbedder`, which
    we might need when defining model parameters.
    """
    def get_output_dim(self) -> int:
        """
        Returns the final output dimension this ``TokenVectorizer`` uses to represent each token.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        from allennlp.experiments.registry  import Registry
        choice = params.pop_choice('type', Registry.list_token_vectorizers())
        return Registry.get_token_vectorizer(choice).from_params(vocab, params)


import torch
from overrides import overrides

from allennlp.common.registrable import Registrable
from allennlp.common.params import Params

class SpanExtractor(torch.nn.Module, Registrable):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractor's define methods for extracting and representing spans
    from a sentence.
    """

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                indicies: torch.LongTensor):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span etc.

        Parameters
        ----------
        sequence_tensor : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        indices : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)``, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the ``sequence_tensor``.

        Returns
        -------
        A tensor of shape ``(batch_size, num_spans, embedded_span_size)``,
        where ``embedded_span_size`` depends on the way spans are represented.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> "SpanExtractor":
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)

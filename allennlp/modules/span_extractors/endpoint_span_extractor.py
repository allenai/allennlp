
import torch
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn.util import batched_index_select, combine_tensors, bucket_values, get_combined_dim
from allennlp.common.checks import ConfigurationError


@SpanExtractor.register("endpoint")
class EndpointSpanExtractor(SpanExtractor):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    The following types of representation are supported, assuming that
    ``x = span_start_embeddings`` and ``y = span_end_embeddings``.

    ``x``, ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give ``x,y,x*y`` as the ``combination`` parameter to this class.
    The computed similarity function would then be ``[x; y; x*y]``, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.
    combination : str, optional (default = "x-y").
        The method used to combine the ``start_embedding`` and ``end_embedding``
        representations. See above for a full description.
    num_width_embeddings : ``int``, optional (default = None).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : ``int``, optional (default = None).
        The embedding size for the span_width features.
    bucket_widths : ``bool``, optional (default = False).
        Whether to bucket the span widths into log-space buckets. If ``False``,
        the raw span widths are used.
    """
    def __init__(self,
                 input_dim: int,
                 combination: str = "x-y",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError("To use a span width embedding representation, you must"
                                     "specify both num_width_buckets and span_width_embedding_dim.")
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                indicies: torch.LongTensor) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in indicies.split(1, dim=-1)]
        start_embeddings = batched_index_select(sequence_tensor, span_starts)
        end_embeddings = batched_index_select(sequence_tensor, span_ends)

        combined_tensors = combine_tensors(self._combination, [start_embeddings, end_embeddings])
        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = bucket_values(span_ends - span_starts,
                                            num_total_buckets=self._num_width_embeddings)
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            return torch.cat([combined_tensors, span_width_embeddings], -1)

        return combined_tensors

    @classmethod
    def from_params(cls, params: Params) -> "EndpointSpanExtractor":
        input_dim = params.pop_int("input_dim")
        combination = params.pop("combination", "x-y")
        num_width_embeddings = params.pop_int("num_width_embeddings", None)
        span_width_embedding_dim = params.pop_int("span_width_embedding_dim", None)
        bucket_widths = params.pop_bool("bucket_widths", False)
        return EndpointSpanExtractor(input_dim=input_dim,
                                     combination=combination,
                                     num_width_embeddings=num_width_embeddings,
                                     span_width_embedding_dim=span_width_embedding_dim,
                                     bucket_widths=bucket_widths)

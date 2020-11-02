from typing import List

import numpy
import torch

from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.nn import util
from allennlp.predictors import Predictor


class SaliencyInterpreter(Registrable):
    """
    A `SaliencyInterpreter` interprets an AllenNLP Predictor's outputs by assigning a saliency
    score to each input token.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        This function finds saliency values for each input token.

        # Parameters

        inputs : `JsonDict`
            The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).

        # Returns

        interpretation : `JsonDict`
            Contains the normalized saliency values for each input token. The dict has entries for
            each instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2:, ... }`.
            Each one of those entries has entries for the saliency of the inputs, e.g.,
            `{grad_input_1: ..., grad_input_2: ... }`.
        """
        raise NotImplementedError("Implement this for saliency interpretations")

    @staticmethod
    def _aggregate_token_embeddings(
        embeddings_list: List[torch.Tensor], token_offsets: List[torch.Tensor]
    ) -> List[numpy.ndarray]:
        if len(token_offsets) == 0:
            return [embeddings.numpy() for embeddings in embeddings_list]
        aggregated_embeddings = []
        # NOTE: This is assuming that embeddings and offsets come in the same order, which may not
        # be true.  But, the intersection of using multiple TextFields with mismatched indexers is
        # currently zero, so we'll delay handling this corner case until it actually causes a
        # problem.  In practice, both of these lists will always be of size one at the moment.
        for embeddings, offsets in zip(embeddings_list, token_offsets):
            span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
            span_mask = span_mask.unsqueeze(-1)
            span_embeddings *= span_mask  # zero out paddings

            span_embeddings_sum = span_embeddings.sum(2)
            span_embeddings_len = span_mask.sum(2)
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

            # All the places where the span length is zero, write in zeros.
            embeddings[(span_embeddings_len == 0).expand(embeddings.shape)] = 0
            aggregated_embeddings.append(embeddings.numpy())
        return aggregated_embeddings

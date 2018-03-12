from overrides import overrides
import torch
from typing import Dict, List, Any

from allennlp.models.ensemble.ensemble import Ensemble
from allennlp.models.archival import load_archive
from allennlp.models import Model
from allennlp.models.reading_comprehension import BidirectionalAttentionFlow
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

@Model.register("bidaf-ensemble")
class BidafEnsemble(Ensemble):

    def __init__(self,
                 submodels: List[BidirectionalAttentionFlow]) -> None:
        super(BidafEnsemble, self).__init__(submodels)

        self.submodels = submodels

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        passage_mask = util.get_text_field_mask(passage).float()

        max_vote = 0
        span_votes = {}
        subresults = []
        for i, submodel in enumerate(self.submodels):
            result = submodel.forward(question, passage, span_start, span_end, metadata)
            subresults.append(result)
            key = (result["span_start_probs"], result["span_end_probs"])
            new_value = span_votes.get(key, 0) + 1
            span_votes[key] = new_value
            if new_value > max_vote:
                max_vote = new_value

        #TODO(micahels): fix float arithmatic

        # Choose the majority-vote span.
        # If there is a tie, break it with the average confidence (span_start_probs + span_end_probs).
        best = 0
        max_average_confidence = 0
        for i, ((span_start_probs, span_end_probs), votes) in enumerate(span_votes.items()):
            print(f"{i}: {votes} vs {max_vote}")
            if votes == max_vote:
                print(i)
                average_confidence = (span_start_probs + span_end_probs) / 2.0
                if average_confidence > max_average_confidence:
                    max_average_confidence = average_confidence
                    best = i

        print("Best:" + best)

        # TODO(michaels): update matrics

        return subresults[best]

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        submodels = []
        paths = params.pop("submodels")
        for path in paths:
           submodels.append(load_archive(path).model)

        return cls(submodels=submodels)

from overrides import overrides
import torch
from typing import Dict, List, Any

from allennlp.models.ensemble.ensemble import Ensemble
from allennlp.models.archival import load_archive
from allennlp.models import Model
from allennlp.models.reading_comprehension import BidirectionalAttentionFlow
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.training.metrics import SquadEmAndF1

@Model.register("bidaf-ensemble")
class BidafEnsemble(Ensemble):

    def __init__(self,
                 submodels: List[BidirectionalAttentionFlow]) -> None:
        super(BidafEnsemble, self).__init__(submodels)

        self.submodels = submodels

        self._squad_metrics = SquadEmAndF1()

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        import copy

        subresults = []
        for i, submodel in enumerate(self.submodels):
            question_copy = {}
            for k, v in question.items():
                question_copy[k] = v.clone()
            passage_copy = {}
            for k, v in passage.items():
                passage_copy[k] = v.clone()
            span_start_copy = span_start.clone()
            span_end_copy = span_end.clone()
            metadata_copy = copy.deepcopy(metadata)
            subresults.append(self.submodels[0].forward(question_copy, passage_copy, span_start_copy, span_end_copy, metadata_copy))

        batch_size = len(subresults[0]["best_span"])

        for i in range(batch_size):
            print(subresults[0]["best_span_str"][i])
            print(subresults[1]["best_span_str"][i])
            print()

        assert subresults[0]["best_span_str"] == subresults[0]["best_span_str"], "impossible"
        assert subresults[0]["best_span_str"] == subresults[1]["best_span_str"], "unexplainable"

        #TODO(michaels): fix float arithmatic
        output = {
            "best_span": torch.zeros(batch_size, 2)
        }
        for batch in range(batch_size):
            max_vote = 0
            span_votes = {}
            for i, subresult in enumerate(subresults):
                key = (subresult["best_span"].data[batch][0], subresult["best_span"].data[batch][1])
                new_value = span_votes.get(key, []) + [i]
                span_votes[key] = new_value
                if len(new_value) > max_vote:
                    max_vote = len(new_value)

            # Choose the majority-vote span.
            # If there is a tie, break it with the average confidence (span_start_probs + span_end_probs).
            best = 0
            max_average_confidence = 0
            for (span_start, span_end), indices in span_votes.items():
                votes = len(indices)
                if votes == max_vote:
                    average_confidence = 0
                    for i in indices:
                        subresult = subresults[i]
                        average = (subresult["span_start_probs"].data[batch][span_start] + subresult["span_end_probs"].data[batch][span_end]) / 2.0
                        if average > average_confidence:
                            average_confidence = average
                    if average_confidence > max_average_confidence:
                        max_average_confidence = average_confidence
                        best = i

            best_span = subresults[best]["best_span"].data[batch]
            output["best_span"][batch] = best_span

            if metadata is not None:
                if not "best_span_str" in output:
                    output["best_span_str"] = []
                best_span_str = subresults[best]["best_span_str"][batch]
                output["best_span_str"].append(best_span_str)

                answer_texts = metadata[batch].get('answer_texts', [])
                if answer_texts:
                    passage_str = metadata[batch]['original_passage']
                    offsets = metadata[batch]['token_offsets'] # character offsets of tokens
                    start_offset = offsets[best_span[0]][0]
                    end_offset = offsets[best_span[1]][1]
                    best_span_string = passage_str[start_offset:end_offset]
                    assert best_span_string == best_span_str, f"{best_span_string} != {best_span_str}"
                    x = subresults[0]["best_span_str"][batch]
                    y = subresults[1]["best_span_str"][batch]
                    print(f"0: {x}")
                    print(f"1: {y}")
                    print(best_span_string)
                    print(answer_texts)
                    print()
                    self._squad_metrics(best_span_string, answer_texts)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'em': exact_match,
            'f1': f1_score,
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        submodels = []
        paths = params.pop("submodels")
        for path in paths:
           submodels.append(load_archive(path).model)

        return cls(submodels=submodels)

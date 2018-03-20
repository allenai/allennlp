from overrides import overrides
from typing import Dict, List, Any
import torch

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

        # Using ModuleList propagates calls to .eval() so dropout is disabled on the submodels in evaluation
        # and prediction.
        self.submodels = torch.nn.ModuleList(submodels)

        self._squad_metrics = SquadEmAndF1()

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        subresults = []
        for submodel in self.submodels:
            subresults.append(submodel.forward(question, passage, span_start, span_end, metadata))

        batch_size = len(subresults[0]["best_span"])

        output = {
                "best_span": torch.zeros(batch_size, 2).long()
        }
        for batch in range(batch_size):
            # Populate span_votes so each key represents a span range that a submodel predicts and the value
            # is the number of models that made the prediction.
            spans = [(subresult["best_span"].data[batch][0], subresult["best_span"].data[batch][1])
                     for subresult in subresults]
            votes: Dict[(int, int), int] = {span:spans.count(span) for span in spans}

            # Choose the majority-vote span.
            # If there is a tie, break it with the average confidence (span_start_probs + span_end_probs).
            options = []
            for i, subresult in enumerate(subresults):
                start = subresult["best_span"].data[batch][0]
                end = subresult["best_span"].data[batch][1]
                num_votes = votes[(start, end)]
                average_confidence = (subresult["span_start_probs"].data[batch][start] +
                                      subresult["span_end_probs"].data[batch][end]) / 2.0
                options.append((-num_votes, -average_confidence, i))

            best = sorted(options)[0][2]
            best_span = subresults[best]["best_span"].data[batch].long()
            output["best_span"][batch] = best_span

            if metadata is not None:
                if "best_span_str" not in output:
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
        assert not vocab, "vocab should be None"

        submodels = []
        paths = params.pop("submodels")
        for path in paths:
            submodels.append(load_archive(path).model)

        return cls(submodels=submodels)

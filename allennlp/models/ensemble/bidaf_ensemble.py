from typing import Dict, List, Any, Tuple

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.models.ensemble.ensemble import Ensemble
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension import BidirectionalAttentionFlow
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.training.metrics import SquadEmAndF1

@Model.register("bidaf-ensemble")
class BidafEnsemble(Ensemble):
    """
    This class ensembles the output from multiple BiDAF models.

    It combines results from the submodels by taking the option with the most votes and breaking ties with
    the average confidence of the start and stop indices.
    """

    def __init__(self,
                 submodels: List[BidirectionalAttentionFlow]) -> None:
        super().__init__(submodels)

        self._squad_metrics = SquadEmAndF1()

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        The forward method runs each of the submodels, then selects the best span from the subresults.
        The best span is the span which most of the submodels predict.  If there is a tie, it is broken
        by the average confidence of the span_start and span_end.

        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of:
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """

        subresults = [submodel(question, passage, span_start, span_end, metadata) for submodel in self.submodels]

        batch_size = len(subresults[0]["best_span"])

        output = {
                "best_span": torch.zeros(batch_size, 2).long(),
                "best_span_str": []
        }
        for index in range(batch_size):
            best_index = ensemble(index, subresults)
            best_span = subresults[best_index]["best_span"].data[index].long()
            output["best_span"][index] = best_span

            if metadata is not None:
                best_span_str = subresults[best_index]["best_span_str"][index]
                output["best_span_str"].append(best_span_str)

                answer_texts = metadata[index].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_str, answer_texts)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'em': exact_match,
                'f1': f1_score,
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        if vocab:
            raise ConfigurationError("vocab should be None")

        submodels = []
        paths = params.pop("submodels")
        for path in paths:
            submodels.append(load_archive(path).model)

        return cls(submodels=submodels)

def ensemble(index: int, subresults: List[Dict[str, torch.Tensor]]) -> int:
    """
    Identifies the best prediction given the results from the submodels.

    Parameters
    ----------
    index : int
        The index within this index to ensemble

    subresults : List[Dict[str, torch.Tensor]]

    Returns
    -------
    The index of the best submodel.
    """

    # Populate span_votes so each key represents a span range that a submodel predicts and the value
    # is the number of models that made the prediction.
    spans = [(subresult["best_span"].data[index][0], subresult["best_span"].data[index][1])
             for subresult in subresults]
    votes: Dict[Tuple[int, int], int] = {span:spans.count(span) for span in spans}

    # Choose the majority-vote span.
    # If there is a tie, break it with the average confidence (span_start_probs + span_end_probs).
    options = []
    for i, subresult in enumerate(subresults):
        start = subresult["best_span"].data[index][0]
        end = subresult["best_span"].data[index][1]
        num_votes = votes[(start, end)]
        average_confidence = (subresult["span_start_probs"].data[index][start] +
                              subresult["span_end_probs"].data[index][end]) / 2.0
        options.append((-num_votes, -average_confidence, i))

    return sorted(options)[0][2]

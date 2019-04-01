from typing import Dict, List, Any

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.models.ensemble import Ensemble
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.training.metrics import SquadEmAndF1

@Model.register("bidaf-ensemble")
class BidafEnsemble(Ensemble):
    """
    This class ensembles the output from multiple BiDAF models.

    It combines results from the submodels by averaging the start and end span probabilities.
    """

    def __init__(self, submodels: List[BidirectionalAttentionFlow]) -> None:
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
        The best span is determined by averaging the probabilities for the start and end of the spans.

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

        best_span = ensemble(subresults)
        output = {
                "best_span": best_span,
                "best_span_str": []
        }
        for index in range(batch_size):
            if metadata is not None:
                passage_str = metadata[index]['original_passage']
                offsets = metadata[index]['token_offsets']
                predicted_span = tuple(best_span[index].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output["best_span_str"].append(best_span_string)

                answer_texts = metadata[index].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'em': exact_match,
                'f1': f1_score,
        }

    # The logic here requires a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidafEnsemble':  # type: ignore
        # pylint: disable=arguments-differ
        if vocab:
            raise ConfigurationError("vocab should be None")

        submodels = []
        paths = params.pop("submodels")
        for path in paths:
            submodels.append(load_archive(path).model)

        return cls(submodels=submodels)

def ensemble(subresults: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Identifies the best prediction given the results from the submodels.

    Parameters
    ----------
    subresults : List[Dict[str, torch.Tensor]]
        Results of each submodel.

    Returns
    -------
    The index of the best submodel.
    """

    # Choose the highest average confidence span.

    span_start_probs = sum(subresult['span_start_probs'] for subresult in subresults) / len(subresults)
    span_end_probs = sum(subresult['span_end_probs'] for subresult in subresults) / len(subresults)
    return get_best_span(span_start_probs.log(), span_end_probs.log()) # type: ignore

from typing import Any, Dict, List, Set, Tuple
from overrides import overrides

import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("mention_recall")
class MentionRecall(Metric):
    def __init__(self) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

    @overrides
    def __call__(
        self,  # type: ignore
        batched_top_spans: torch.Tensor,
        batched_metadata: List[Dict[str, Any]],
    ):
        for top_spans, metadata in zip(batched_top_spans.tolist(), batched_metadata):

            gold_mentions: Set[Tuple[int, int]] = {
                mention for cluster in metadata["clusters"] for mention in cluster
            }
            predicted_spans: Set[Tuple[int, int]] = {(span[0], span[1]) for span in top_spans}
            self._num_gold_mentions += len(gold_mentions)
            self._num_recalled_mentions += len(gold_mentions & predicted_spans)

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions / self._num_gold_mentions
        if reset:
            self.reset()
        return recall

    @overrides
    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

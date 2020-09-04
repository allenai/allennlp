from typing import Any, Dict, List, Tuple

from allennlp.common.registrable import Registrable


class MultiTaskScheduler(Registrable):
    """
    TODO
    """

    def get_task_proportions(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_batch_ordering(self) -> List[Tuple[str, int]]:
        raise NotImplementedError

    def set_epoch_metrics(self, epoch_metrics: Dict[str, Any]) -> None:
        raise NotImplementedError

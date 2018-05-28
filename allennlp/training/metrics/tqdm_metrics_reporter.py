from typing import Dict, Iterable


class TqdmMetricsReporter(Iterable):

    def __init__(self, tqdm):
        super().__init__()
        self._tqdm = tqdm

    def report(self,
               metrics: Dict[str, float])-> None:
        desc = self._description_from_metrics(metrics)
        self._tqdm.set_description(desc, refresh=False)

    @classmethod
    def _description_from_metrics(cls, metrics: Dict[str, float]) -> str:
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def __iter__(self):
        return self._tqdm.__iter__()

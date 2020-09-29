from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    run_distributed_test,
    global_distributed_metric,
)
from allennlp.training.metrics import Average


class AverageTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.metric = Average()

    @multi_device
    def test_distributed_average(self, device: str):
        device_ids = [-1, -1] if device == "cpu" else [0, 1]
        metric_kwargs = {
            "value": [1.0, 2.0],
        }
        run_distributed_test(
            device_ids,
            global_distributed_metric,
            self.metric,
            metric_kwargs,
            1.5,
            exact=True,
        )

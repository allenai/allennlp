from typing import Iterator, List, Dict

import torch
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.data_loaders import TensorDict
from allennlp.models import Model
from allennlp.evaluation import Evaluator
from allennlp.common import Params


class DummyDataLoader:
    def __init__(self, outputs: List[TensorDict]) -> None:
        super().__init__()
        self._outputs = outputs

    def __iter__(self) -> Iterator[TensorDict]:
        yield from self._outputs

    def __len__(self):
        return len(self._outputs)

    def set_target_device(self, _):
        pass


class DummyModel(Model):
    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore
        return kwargs


class TestEvaluator(AllenNlpTestCase):
    def setup_method(self):
        self.evaluator = Evaluator.from_params(Params({"batch_postprocessor": "simple"}))

    def test_evaluate_calculates_average_loss(self):
        losses = [7.0, 9.0, 8.0]
        outputs = [{"loss": torch.Tensor([loss])} for loss in losses]
        data_loader = DummyDataLoader(outputs)
        metrics = self.evaluator(DummyModel(), data_loader, "")  # type: ignore
        assert metrics["loss"] == pytest.approx(8.0)

    def test_evaluate_calculates_average_loss_with_weights(self):
        losses = [7.0, 9.0, 8.0]
        weights = [10, 2, 1.5]
        inputs = zip(losses, weights)
        outputs = [
            {"loss": torch.Tensor([loss]), "batch_weight": torch.Tensor([weight])}
            for loss, weight in inputs
        ]
        data_loader = DummyDataLoader(outputs)
        metrics = self.evaluator(DummyModel(), data_loader, "batch_weight")  # type: ignore
        assert metrics["loss"] == pytest.approx((70 + 18 + 12) / 13.5)

    def test_to_params(self):
        assert self.evaluator.to_params() == {
            "type": "simple",
            "cuda_device": -1,
            "batch_postprocessor": {"type": "simple"},
        }

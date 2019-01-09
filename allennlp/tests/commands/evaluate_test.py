# pylint: disable=invalid-name,no-self-use
import argparse
import json
import torch
from typing import Iterator, List, Dict

from flaky import flaky

from allennlp.commands.evaluate import evaluate_from_args, Evaluate, evaluate
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DataIterator
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models import Model


class DummyIterator(DataIterator):
    def __init__(self, count: int):
        super().__init__()
        self._count = count

    def __call__(self, *args, **kwargs) -> Iterator[TensorDict]:
        while self._count > 0:
            self._count -= 1
            yield {}


class DummyModel(Model):
    def __init__(self, outputs: List):
        super().__init__(None) # type: ignore
        self._outputs = outputs

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        return self._outputs.pop(0)


class TestEvaluate(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title='Commands', metavar='')
        Evaluate().add_subparser('evaluate', subparsers)

    def test_evaluate_calculates_average_loss(self):
        losses = [7.0, 9.0, 8.0]
        outputs = [{"loss": torch.tensor(loss)} for loss in losses]
        model = DummyModel(outputs)
        iterator = DummyIterator(len(outputs))
        metrics = evaluate(model, None, iterator, -1, "")
        self.assertAlmostEqual(metrics["loss"], 8.0)

    def test_evaluate_calculates_average_loss_with_weights(self):
        losses = [7.0, 9.0, 8.0]
        weights = [10, 2, 1.5]
        inputs = zip(losses, weights)
        outputs = [{"loss": torch.tensor(loss), "batch_weight": torch.tensor(weight)} for loss, weight in inputs]
        model = DummyModel(outputs)
        iterator = DummyIterator(len(outputs))
        metrics = evaluate(model, None, iterator, -1, "batch_weight")
        self.assertAlmostEqual(metrics["loss"], (70 + 18 + 12)/13.5)

    @flaky
    def test_evaluate_from_args(self):
        kebab_args = ["evaluate", str(self.FIXTURES_ROOT / "bidaf" / "serialization" / "model.tar.gz"),
                      str(self.FIXTURES_ROOT / "data" / "squad.json"),
                      "--cuda-device", "-1"]

        args = self.parser.parse_args(kebab_args)
        metrics = evaluate_from_args(args)
        assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1', 'loss'}

    def test_output_file_evaluate_from_args(self):
        output_file = str(self.TEST_DIR / "metrics.json")
        kebab_args = ["evaluate", str(self.FIXTURES_ROOT / "bidaf" / "serialization" / "model.tar.gz"),
                      str(self.FIXTURES_ROOT / "data" / "squad.json"),
                      "--cuda-device", "-1",
                      "--output-file", output_file]
        args = self.parser.parse_args(kebab_args)
        computed_metrics = evaluate_from_args(args)
        with open(output_file, 'r') as file:
            saved_metrics = json.load(file)
        assert computed_metrics == saved_metrics

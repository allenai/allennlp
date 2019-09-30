import argparse
import json
from typing import Iterator, List, Dict, Iterable

import torch
from flaky import flaky

from allennlp.commands.evaluate import evaluate_from_args, Evaluate, evaluate
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DataIterator, Instance
from allennlp.data.dataset import Batch
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models import Model


class DummyIterator(DataIterator):
    def __init__(self, outputs: List[TensorDict]) -> None:
        super().__init__()
        self._outputs = outputs

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:
        yield from self._outputs

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        raise NotImplementedError


class DummyModel(Model):
    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore
        return kwargs


class TestEvaluate(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title='Commands', metavar='')
        Evaluate().add_subparser('evaluate', subparsers)

    def test_evaluate_calculates_average_loss(self):
        losses = [7.0, 9.0, 8.0]
        outputs = [{"loss": torch.Tensor([loss])} for loss in losses]
        iterator = DummyIterator(outputs)
        metrics = evaluate(DummyModel(), None, iterator, -1, "")
        self.assertAlmostEqual(metrics["loss"], 8.0)

    def test_evaluate_calculates_average_loss_with_weights(self):
        losses = [7.0, 9.0, 8.0]
        weights = [10, 2, 1.5]
        inputs = zip(losses, weights)
        outputs = [{"loss": torch.Tensor([loss]), "batch_weight": torch.Tensor([weight])}
                   for loss, weight in inputs]
        iterator = DummyIterator(outputs)
        metrics = evaluate(DummyModel(), None, iterator, -1, "batch_weight")
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

    def test_evaluate_works_with_vocab_expansion(self):
        archive_path = str(self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz")
        # snli2 has a extra token ("seahorse") in it.
        evaluate_data_path = str(self.FIXTURES_ROOT / 'data' / 'snli2.jsonl')
        embeddings_filename = str(self.FIXTURES_ROOT / 'data' / 'seahorse_embeddings.gz')  # has only seahorse vector
        embedding_sources_mapping = json.dumps({"_text_field_embedder.token_embedder_tokens": embeddings_filename})
        kebab_args = ["evaluate", archive_path, evaluate_data_path, "--cuda-device", "-1"]

        # Evaluate 1 with no vocab expansion,
        # Evaluate 2 with vocab expansion with no pretrained embedding file.
        # Evaluate 3 with vocab expansion with given pretrained embedding file.
        metrics_1 = evaluate_from_args(self.parser.parse_args(kebab_args))
        metrics_2 = evaluate_from_args(self.parser.parse_args(kebab_args + ["--extend-vocab"]))
        metrics_3 = evaluate_from_args(self.parser.parse_args(kebab_args + ["--embedding-sources-mapping",
                                                                            embedding_sources_mapping]))
        assert metrics_1 != metrics_2
        assert metrics_2 != metrics_3

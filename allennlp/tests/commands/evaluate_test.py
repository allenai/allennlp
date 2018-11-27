# pylint: disable=invalid-name,no-self-use
import argparse
import json

from flaky import flaky

from allennlp.commands.evaluate import evaluate_from_args, Evaluate
from allennlp.common.testing import AllenNlpTestCase


class TestEvaluate(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title='Commands', metavar='')
        Evaluate().add_subparser('evaluate', subparsers)

    @flaky
    def test_evaluate_from_args(self):
        kebab_args = ["evaluate", str(self.FIXTURES_ROOT / "bidaf" / "serialization" / "model.tar.gz"),
                      str(self.FIXTURES_ROOT / "data" / "squad.json"),
                      "--cuda-device", "-1"]

        args = self.parser.parse_args(kebab_args)
        metrics = evaluate_from_args(args)
        assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1'}

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

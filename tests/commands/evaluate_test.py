# pylint: disable=invalid-name,no-self-use
import argparse

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
        kebab_args = ["evaluate", "tests/fixtures/bidaf/serialization/model.tar.gz",
                      "--evaluation-data-file", "tests/fixtures/data/squad.json",
                      "--cuda-device", "-1"]

        args = self.parser.parse_args(kebab_args)
        metrics = evaluate_from_args(args)
        assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1'}

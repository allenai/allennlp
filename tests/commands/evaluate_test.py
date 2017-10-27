# pylint: disable=invalid-name,no-self-use
import argparse

from flaky import flaky

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.evaluate import evaluate_from_args, Evaluate


class TestEvaluate(AllenNlpTestCase):

    @flaky
    def test_evaluate_from_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Evaluate().add_subparser('evaluate', subparsers)

        raw_args = ["evaluate",
                    "--archive_file", "tests/fixtures/bidaf/serialization/model.tar.gz",
                    "--evaluation_data_file", "tests/fixtures/data/squad.json"]

        args = parser.parse_args(raw_args)

        metrics = evaluate_from_args(args)

        assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1'}

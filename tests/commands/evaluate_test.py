# pylint: disable=invalid-name,no-self-use
import argparse

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.evaluate import evaluate_from_args, add_subparser


class TestEvaluate(AllenNlpTestCase):

    def test_evaluate_from_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers)

        raw_args = ["evaluate",
                    "--config_file", "tests/fixtures/bidaf/experiment.json",
                    "--evaluation_data_file", "tests/fixtures/data/squad.json"]

        args = parser.parse_args(raw_args)

        metrics = evaluate_from_args(args)

        assert metrics == {'full_span_acc': 0.0, 'span_end_acc': 0.0, 'span_start_acc': 0.0}

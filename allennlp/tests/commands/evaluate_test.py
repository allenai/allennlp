# pylint: disable=invalid-name,no-self-use
import argparse
from io import IOBase
from flaky import flaky
from allennlp.common.tqdm import Tqdm
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
                      "--evaluation-data-file", str(self.FIXTURES_ROOT / "data" / "squad.json"),
                      "--cuda-device", "-1"]

        args = self.parser.parse_args(kebab_args)
        metrics = evaluate_from_args(args)
        assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1'}

    @flaky
    def test_progress_logs_are_correct(self):
        dummy_logger = DummyLogger("start_acc: 0.0000, end_acc: 0.0000, "
                                   "span_acc: 0.0000, em: 0.0000, f1: 0.1524, loss: 0.0000 ||: 100%|##########|")
        Tqdm.set_io_output(dummy_logger)
        kebab_args = ["evaluate", str(self.FIXTURES_ROOT / "bidaf" / "serialization" / "model.tar.gz"),
                      "--evaluation-data-file", str(self.FIXTURES_ROOT / "data" / "squad.json"),
                      "--cuda-device", "-1"]
        args = self.parser.parse_args(kebab_args)
        evaluate_from_args(args)
        assert dummy_logger.seen_message


class DummyLogger(IOBase):
    """
    Used to check that the expected string is sent to tqdm
    """
    def __init__(self, expected_output: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_output = expected_output
        self.seen_message = False

    def write(self, message: str):
        self.seen_message = self.seen_message or message.__contains__(self._expected_output)

    def flush(self):
        pass

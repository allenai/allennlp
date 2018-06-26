import atexit
import logging
import os
import pathlib
import subprocess

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

ABBREVIATIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"
SEMPRE_DIR = str(pathlib.Path('data/'))
SEMPRE_ABBREVIATIONS_PATH = os.path.join(SEMPRE_DIR, "abbreviations.tsv")
SEMPRE_GRAMMAR_PATH = os.path.join(SEMPRE_DIR, "grow.grammar")


class RateCalculusAccuracy(Metric):
    def __init__(self, table_directory: str) -> None:
        self._count = 0
        self._correct = 0

    @overrides
    def __call__(self, logical_form: str, example_lisp_string: str):  # type: ignore
        """
        Parameters
        ----------
        example_lisp_string : ``str``
            The value to average.
        """
        denotation_correct = self.evaluate_logical_form(logical_form, example_lisp_string)
        if denotation_correct:
            self._correct += 1
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        accuracy = self._correct / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._count = 0
        self._correct = 0

    def __str__(self):
        return f"RateCalculusAccuracy(correct={self._correct}, count={self._count})"

    def evaluate_logical_form(self, logical_form: str, example_lisp_string: str) -> bool:
        print("LOGICAL FORM in evaluate: ", logical_form)
        if not logical_form or logical_form.startswith('Error'):
            return False
        return True

import atexit
import logging
import os
import subprocess

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SEMPRE_EXECUTOR_JAR = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-executor-0.1.0.jar"
ABBREVIATIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"
SEMPRE_DIR = 'data/'


class WikiTablesAccuracy(Metric):
    def __init__(self, table_directory: str) -> None:
        self._table_directory = table_directory
        self._executor_process: subprocess.Popen = None
        self._create_sempre_executor()
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
        return f"WikiTablesAccuracy(correct={self._correct}, count={self._count})"

    def evaluate_logical_form(self, logical_form: str, example_lisp_string: str) -> bool:
        if not logical_form or logical_form.startswith('Error'):
            return False
        if example_lisp_string[-1] != '\n':
            example_lisp_string += '\n'
        if logical_form[-1] != '\n':
            logical_form += '\n'
        self._executor_process.stdin.write(example_lisp_string.encode('utf-8'))
        self._executor_process.stdin.write(logical_form.encode('utf-8'))
        self._executor_process.stdin.flush()
        result = self._executor_process.stdout.readline().decode().strip()
        return result == '1.0'

    def _create_sempre_executor(self) -> None:
        """
        Creates a server running SEMPRE that we can send logical forms to for evaluation.  This
        uses inter-process communication, because SEMPRE is java code.  We also need to be careful
        to clean up the process when our program exits.
        """
        if self._executor_process:
            return

        # It'd be much nicer to just use `cached_path` for these files.  However, the SEMPRE jar
        # that we're using expects to find these files in a particular location, so we need to make
        # sure we put the files in that location.
        os.makedirs(SEMPRE_DIR, exist_ok=True)
        abbreviations_path = os.path.join(SEMPRE_DIR, 'abbreviations.tsv')
        if not os.path.exists(abbreviations_path):
            subprocess.run(f'wget {ABBREVIATIONS_FILE}', shell=True)
            subprocess.run(f'mv wikitables-abbreviations.tsv {abbreviations_path}', shell=True)

        grammar_path = os.path.join(SEMPRE_DIR, 'grow.grammar')
        if not os.path.exists(grammar_path):
            subprocess.run(f'wget {GROW_FILE}', shell=True)
            subprocess.run(f'mv wikitables-grow.grammar {grammar_path}', shell=True)

        args = ['java', '-jar', cached_path(SEMPRE_EXECUTOR_JAR), 'serve', self._table_directory]
        self._executor_process = subprocess.Popen(args,
                                                  stdin=subprocess.PIPE,
                                                  stdout=subprocess.PIPE,
                                                  bufsize=1)

        lines = []
        for _ in range(6):
            # SEMPRE outputs six lines of stuff when it loads that I can't disable.  So, we clear
            # that here.
            lines.append(str(self._executor_process.stdout.readline()))
        assert 'Parser' in lines[-1], "SEMPRE server output unexpected; the server may have changed"
        logger.info("Started SEMPRE server for evaluating logical forms")

        # This is supposed to ensure that the subprocess gets killed when python exits.
        atexit.register(self._stop_sempre_executor)

    def _stop_sempre_executor(self) -> None:
        if not self._executor_process:
            return
        self._executor_process.terminate()
        self._executor_process = None
        logger.info("Stopped SEMPRE server")

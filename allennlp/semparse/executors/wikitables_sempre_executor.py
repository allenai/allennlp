import atexit
import logging
import os
import pathlib
import shutil
import subprocess
import requests

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import check_for_java

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SEMPRE_EXECUTOR_JAR = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-executor-0.1.0.jar"
ABBREVIATIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"
SEMPRE_DIR = str(pathlib.Path('data/'))
SEMPRE_ABBREVIATIONS_PATH = os.path.join(SEMPRE_DIR, "abbreviations.tsv")
SEMPRE_GRAMMAR_PATH = os.path.join(SEMPRE_DIR, "grow.grammar")


class WikiTablesSempreExecutor:
    """
    This class evaluates lambda-DCS logical forms by calling out to SEMPRE, where the particular
    lambda-DCS language we use was defined.  It's a huge pain to have to rely on a call to a java
    subprocess, but it's way easier than trying to write our own executor for the language.

    Because of how the SEMPRE executor works, we need to have access to the original table on disk,
    and we need to pass in the full lisp "example" string that's given in the dataset.  SEMPRE
    parses the example string (which includes a path to the table), reads the table, executes the
    logical form in the context of the table, and compares the answer produced to the answer
    specified in the example string.

    We don't even get back the denotation of the logical form here, because then we'd have to do a
    comparison with the correct answer, and that's a bit messy - better to just let SEMPRE do it.
    This is why we only provide a :func:`evaluate_logical_form` method that returns a ``bool``
    instead of an ``execute`` method returning an answer.  You might think that if we got the
    answer back, we could at least use this for a demo.  The sad thing is that even that doesn't
    work, because this executor relies on having the table for the example already accessible on
    disk, which we don't have in the case of a demo - we have to do extra stuff there to get it to
    work, including writing the table to disk so that SEMPRE can access it!  It's all a bit of a
    mess.
    """
    def __init__(self, table_directory: str) -> None:
        self._table_directory = table_directory
        self._executor_process: subprocess.Popen = None
        self._should_remove_sempre_dir = not os.path.exists(SEMPRE_DIR)
        self._create_sempre_executor()

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
            result = requests.get(ABBREVIATIONS_FILE)
            with open(abbreviations_path, 'wb') as downloaded_file:
                downloaded_file.write(result.content)

        grammar_path = os.path.join(SEMPRE_DIR, 'grow.grammar')
        if not os.path.exists(grammar_path):
            result = requests.get(GROW_FILE)
            with open(grammar_path, 'wb') as downloaded_file:
                downloaded_file.write(result.content)

        if not check_for_java():
            raise RuntimeError('Java is not installed properly.')
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
        if self._should_remove_sempre_dir and os.path.exists(SEMPRE_DIR):
            shutil.rmtree(SEMPRE_DIR)
            logger.info(f"Removed SEMPRE data directory ({SEMPRE_DIR})")

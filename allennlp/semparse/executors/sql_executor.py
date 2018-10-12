import logging
from typing import List

import sqlite3
import multiprocessing
from multiprocessing import Process
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
MULTIPROCESSING_LOGGER = multiprocessing.get_logger()

class SqlExecutor:
    """
    This class evaluates SQL queries by connecting to a SQLite database. Because SQLite is disk-based
    we just need to provide one file with the location. We execute the predicted SQL query and the labeled
    queries against the database and check if they execute to the same table.
    """

    def __init__(self, database_file: str) -> None:
        # Initialize a cursor to our sqlite database, so we can execute SQL queries for denotation accuracy.
        self._database_file = cached_path(database_file)
        self._connection = sqlite3.connect(self._database_file)
        self._cursor = self._connection.cursor()

    def evaluate_sql_query(self,
                           predicted_sql_query: str,
                           sql_query_labels: List[str]) -> int:
        # We set the logging level for the subprocesses to warning, otherwise, it will
        # log every time a process starts and stops.
        MULTIPROCESSING_LOGGER.setLevel(logging.WARNING)

        # Since the query might hang, we run in another process and kill it if it
        # takes too long.
        process = Process(target=self._evaluate_sql_query_subprocess,
                          args=(predicted_sql_query, sql_query_labels))
        process.start()

        # If the query has not finished in 3 seconds then we will proceed.
        process.join(3)
        denotation_correct = process.exitcode # type: ignore

        if process.is_alive():
            logger.warning("Evaluating query took over 3 seconds, skipping query")
            process.terminate()
            process.join()

        if denotation_correct is None:
            denotation_correct = 0

        return denotation_correct

    def _evaluate_sql_query_subprocess(self, predicted_query: str, sql_query_labels: List[str]) -> int:
        """
        We evaluate here whether the predicted query and the query label evaluate to the
        exact same table. This method is only called by the subprocess, so we just exit with
        1 if it is correct and 0 otherwise.
        """

        postprocessed_predicted_query = self.postprocess_query_sqlite(predicted_query)

        try:
            self._cursor.execute(postprocessed_predicted_query)
            predicted_rows = self._cursor.fetchall()
        except sqlite3.Error as error:
            logger.warning(f'Error executing predicted: {error}')
            exit(0)

        # If predicted table matches any of the reference tables then it is counted as correct.
        target_rows = None
        for sql_query_label in sql_query_labels:
            postprocessed_sql_query_label = self.postprocess_query_sqlite(sql_query_label)
            try:
                self._cursor.execute(postprocessed_sql_query_label)
                target_rows = self._cursor.fetchall()
            except sqlite3.Error as error:
                logger.warning(f'Error executing predicted: {error}')
            if predicted_rows == target_rows:
                exit(1)
        exit(0)

    @staticmethod
    def postprocess_query_sqlite(query: str):
        # The dialect of SQL that SQLite takes is not exactly the same as the labeled data.
        # We strip off the parentheses that surround the entire query here.
        query = query.strip()
        if query.startswith('('):
            return query[1:query.rfind(')')] + ';'
        return query

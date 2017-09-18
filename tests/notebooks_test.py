import os

import nbformat
from nbconvert.preprocessors.execute import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor

from allennlp.common.testing import AllenNlpTestCase

class TestNotebooks(AllenNlpTestCase):
    def test_vocabulary_tutorial(self):
        assert self.execute_notebook("tutorials/notebooks/vocabulary.ipynb")

    def test_data_pipeline_tutorial(self):
        assert self.execute_notebook("tutorials/notebooks/data_pipeline.ipynb")

    def test_embedding_tokens_tutorial(self):
        assert self.execute_notebook("tutorials/notebooks/embedding_tokens.ipynb")

    @staticmethod
    def execute_notebook(notebook_path: str):
        with open(notebook_path, encoding='utf-8') as notebook:
            contents = nbformat.read(notebook, as_version=4)

        execution_processor = ExecutePreprocessor(timeout=60, kernel_name="python3")
        try:
            # Actually execute the notebook in the current working directory.
            execution_processor.preprocess(contents, {'metadata': {'path': os.getcwd()}})
            return True
        except CellExecutionError:
            # This is a big chunk of JSON, but the stack trace makes it reasonably
            # clear which cell the error occurred in, so fixing it by actually
            # running the notebook will probably be easier.
            print(contents)
            return False

# pylint: disable=no-self-use,invalid-name
import os
import pathlib
import sys

import torch

from allennlp.common import util
from allennlp.common.testing import AllenNlpTestCase


class TestCommonUtils(AllenNlpTestCase):
    def test_group_by_count(self):
        assert util.group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 20) == [[1, 2, 3], [4, 5, 6], [7, 20, 20]]

    def test_pad_sequence_to_length(self):
        assert util.pad_sequence_to_length([1, 2, 3], 5) == [1, 2, 3, 0, 0]
        assert util.pad_sequence_to_length([1, 2, 3], 5, default_value=lambda: 2) == [1, 2, 3, 2, 2]
        assert util.pad_sequence_to_length([1, 2, 3], 5, padding_on_right=False) == [0, 0, 1, 2, 3]

    def test_namespace_match(self):
        assert util.namespace_match("*tags", "tags")
        assert util.namespace_match("*tags", "passage_tags")
        assert util.namespace_match("*tags", "question_tags")
        assert util.namespace_match("tokens", "tokens")
        assert not util.namespace_match("tokens", "stemmed_tokens")

    def test_sanitize(self):
        assert util.sanitize(torch.Tensor([1, 2])) == [1, 2]
        assert util.sanitize(torch.LongTensor([1, 2])) == [1, 2]

    def test_import_submodules(self):
        os.makedirs(os.path.join(self.TEST_DIR, 'mymodule'))
        pathlib.Path(os.path.join(self.TEST_DIR, 'mymodule/__init__.py')).touch()
        os.makedirs(os.path.join(self.TEST_DIR, 'mymodule/submodule'))
        pathlib.Path(os.path.join(self.TEST_DIR, 'mymodule/submodule/__init__.py')).touch()

        sys.path.insert(0, self.TEST_DIR)

        assert 'mymodule' not in sys.modules
        assert 'mymodule.submodule' not in sys.modules

        util.import_submodules('mymodule')

        assert 'mymodule' in sys.modules
        assert 'mymodule.submodule' in sys.modules

        sys.path.remove(self.TEST_DIR)

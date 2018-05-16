# pylint: disable=invalid-name,no-self-use
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.test_install import _get_project_root


class TestTestInstall(AllenNlpTestCase):
    def test_get_project_root(self):
        project_root = _get_project_root()
        assert os.path.exists(os.path.join(project_root, "tests"))
        assert os.path.exists(os.path.join(project_root, "tests",
                                           "commands", "test_install_test.py"))

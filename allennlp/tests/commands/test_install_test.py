# pylint: disable=invalid-name,no-self-use
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.test_install import _get_module_root


class TestTestInstall(AllenNlpTestCase):
    def test_get_module_root(self):
        project_root = _get_module_root()
        assert os.path.exists(os.path.join(project_root, "tests"))
        assert os.path.exists(os.path.join(project_root, "run.py"))

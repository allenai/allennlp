import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.test_install import _get_module_root


class TestTestInstall(AllenNlpTestCase):
    def test_get_module_root(self):
        """
        When a user runs `allennlp test-install`, we have no idea where
        they're running it from, so we do an `os.chdir` to the _module_
        root in order to get all the paths in the fixtures to resolve properly.

        The logic within `allennlp test-install` is pretty hard to test in
        its entirety, so this test is verifies that the `os.chdir` component
        works properly by checking that we correctly find the path to
        `os.chdir` to.
        """
        project_root = _get_module_root()
        assert os.path.exists(os.path.join(project_root, "__main__.py"))

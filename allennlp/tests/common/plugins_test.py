import distutils.dir_util
import tempfile

import pytest
from overrides import overrides

from allennlp.commands import Subcommand
from allennlp.common.plugins import (
    discover_file_plugins,
    discover_namespace_plugins,
    discover_plugins,
    import_plugins,
)
from allennlp.common.testing import AllenNlpTestCase
from allennlp.tests.common.plugins_util import push_python_project


class TestPlugins(AllenNlpTestCase):
    @overrides
    def setUp(self):
        super().setUp()
        self.plugins_root = self.FIXTURES_ROOT / "plugins"
        # "a" sets a "global" namespace plugin, because it's gonna be installed with pip.
        self.project_a_fixtures_root = self.plugins_root / "project_a"
        # "b" sets a "local" namespace plugin, because it's supposed to be run from that directory.
        self.project_b_fixtures_root = self.plugins_root / "project_b"
        # "c" sets a "global" namespace plugin, because it's gonna be installed with pip.
        self.project_c_fixtures_root = self.plugins_root / "project_c"
        # "d" sets a "local" file plugin, because it's supposed to be run from that directory
        # and has a ".allennlp_plugins" file in it.
        self.project_d_fixtures_root = self.plugins_root / "project_d"

    def test_no_plugins(self):
        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

    def test_namespace_package_does_not_exist(self):
        available_plugins = set(discover_namespace_plugins("dummy_namespace"))
        self.assertSetEqual(set(), available_plugins)

    def test_file_plugins_does_not_exist(self):
        available_plugins = set(discover_file_plugins("dummy_file"))
        self.assertSetEqual(set(), available_plugins)
    def test_local_namespace_plugin(self):
        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

        with push_python_project(self.project_b_fixtures_root):
            available_plugins = set(discover_plugins())
            self.assertSetEqual({"allennlp_plugins.b"}, available_plugins)

            import_plugins()
            subcommands_available = Subcommand.list_available()
            self.assertIn("b", subcommands_available)

    def test_file_plugin(self):
        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

        with push_python_project(self.project_d_fixtures_root):
            available_plugins = set(discover_plugins())
            self.assertSetEqual({"d"}, available_plugins)

            import_plugins()
            subcommands_available = Subcommand.list_available()
            self.assertIn("d", subcommands_available)

    def test_local_namespace_plugin_different_path(self):
        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

        with tempfile.TemporaryDirectory() as temp_dir_b:
            distutils.dir_util.copy_tree(self.project_b_fixtures_root, temp_dir_b)

            # We move to another directory with a different plugin "b", as if it were another separate project
            # which is not installed ("local" usage of the plugin declared in the namespace).
            with push_python_project(temp_dir_b):
                available_plugins = set(discover_plugins())
                self.assertSetEqual({"allennlp_plugins.b"}, available_plugins)

                import_plugins()
                subcommands_available = Subcommand.list_available()
                self.assertIn("b", subcommands_available)

import contextlib
import distutils.dir_util
import tempfile

import pytest
from overrides import overrides
from pip._internal.cli.main import main as pip_main

from allennlp.commands import Subcommand
from allennlp.common.plugins import (
    discover_file_plugins,
    discover_namespace_plugins,
    discover_plugins,
    import_plugins,
)
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ContextManagerFunctionReturnType, PathType, push_python_path, pushd


@contextlib.contextmanager
def pip_install(path: PathType, package_name: str) -> ContextManagerFunctionReturnType[None]:
    pip_main(["install", str(path)])
    try:
        yield
    finally:
        pip_main(["uninstall", "-y", package_name])


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
        available_plugins = list(discover_plugins())
        self.assertEqual(0, len(available_plugins))

    def test_namespace_package_does_not_exist(self):
        available_plugins = list(discover_namespace_plugins("dummy_namespace"))
        self.assertEqual(0, len(available_plugins))

    def test_file_plugins_does_not_exist(self):
        available_plugins = list(discover_file_plugins("dummy_file"))
        self.assertEqual(0, len(available_plugins))

    def test_namespace_plugins_are_discovered_and_imported(self):
        # We make plugins "a" and "c" available as packages, each from other directories, as if they were
        # separate installed projects ("global" usage of the plugins).
        # We move to another directory with a different plugin "b", as if it were another separate project
        # which is not installed ("local" usage of the plugin declared in the namespace).
        # In general when we run scripts or commands in a project, the current directory is the root of it
        # and is part of the path. So we emulate this here with `push_python_path`.
        with pip_install(self.project_a_fixtures_root, "a"), pip_install(
            self.project_c_fixtures_root, "c"
        ), pushd(self.project_b_fixtures_root), push_python_path("."):
            available_plugins = list(discover_plugins())
            self.assertEqual(3, len(available_plugins))

            import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertIn("A", subcommands_available)
            self.assertIn("B", subcommands_available)
            self.assertIn("C", subcommands_available)

    def test_local_namespace_plugin_works_from_a_completely_different_path(self):
        with tempfile.TemporaryDirectory() as temp_dir_b:
            distutils.dir_util.copy_tree(self.project_b_fixtures_root, temp_dir_b)

            # We move to another directory with a different plugin "b", as if it were another separate project
            # which is not installed ("local" usage of the plugin declared in the namespace).
            # In general when we run scripts or commands in a project, the current directory is the root of it
            # and is part of the path. So we emulate this here with `push_python_path`.
            with pushd(temp_dir_b), push_python_path("."):
                available_plugins = list(discover_plugins())
                self.assertEqual(1, len(available_plugins))

                import_plugins()
                subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
                self.assertIn("B", subcommands_available)

    def test_namespace_and_file_plugins_are_discovered_and_imported(self):
        # We make plugins "a" and "c" available as packages, each from other directories, as if they were
        # separate installed projects ("global" usage of the plugins).
        # We move to another directory with a different plugin "b", as if it were another separate project
        # which is not installed ("local" usage of the plugin declared in a file).
        # In general when we run scripts or commands in a project, the current directory is the root of it
        # and is part of the path. So we emulate this here with `push_python_path`.
        with pip_install(self.project_a_fixtures_root, "a"), pip_install(
            self.project_c_fixtures_root, "c"
        ), pushd(self.project_d_fixtures_root), push_python_path("."):
            available_plugins = list(discover_plugins())
            self.assertEqual(3, len(available_plugins))

            import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertIn("A", subcommands_available)
            self.assertIn("C", subcommands_available)
            self.assertIn("D", subcommands_available)

    def test_reload_plugins_add_new_plugin(self):
        with pip_install(self.project_a_fixtures_root, "a"):
            available_plugins = list(discover_plugins())
            self.assertEqual(1, len(available_plugins))

            import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertIn("A", subcommands_available)

            with pip_install(self.project_c_fixtures_root, "c"):
                available_plugins = list(discover_plugins())
                self.assertEqual(2, len(available_plugins))

                import_plugins()
                subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
                self.assertIn("A", subcommands_available)
                self.assertIn("C", subcommands_available)

    @pytest.mark.skip("Plugin unloading is not supported.")
    def test_unload_plugin(self):
        with pip_install(self.project_a_fixtures_root, "a"):
            available_plugins = list(discover_plugins())
            self.assertEqual(1, len(available_plugins))

            import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertIn("A", subcommands_available)

        available_plugins = list(discover_plugins())
        self.assertEqual(0, len(available_plugins))

        import_plugins()
        subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
        self.assertNotIn("A", subcommands_available)

    @pytest.mark.skip("Plugin unloading is not supported.")
    def test_reload_plugins_removes_one_adds_one(self):
        with pip_install(self.project_a_fixtures_root, "a"):
            available_plugins = list(discover_plugins())
            self.assertEqual(1, len(available_plugins))

            import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertIn("A", subcommands_available)
            self.assertNotIn("C", subcommands_available)

        with pip_install(self.project_c_fixtures_root, "c"):
            available_plugins = list(discover_plugins())
            self.assertEqual(1, len(available_plugins))

            import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertNotIn("A", subcommands_available)
            self.assertIn("C", subcommands_available)

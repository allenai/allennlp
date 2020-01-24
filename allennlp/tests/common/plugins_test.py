import contextlib
import distutils.dir_util
import tempfile

from overrides import overrides
from pip._internal.cli.main import main as pip_main

import allennlp
from allennlp.commands import Subcommand
from allennlp.common.plugins import discover_plugins
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ContextManagerFunctionReturnType, PathType, push_python_path, pushd


@contextlib.contextmanager
def pip_install(path: PathType, package_name: str) -> ContextManagerFunctionReturnType[None]:
    pip_main(["install", str(path)])
    try:
        yield
    finally:
        pip_main(["uninstall", "-y", package_name])


class TestCommonUtils(AllenNlpTestCase):
    @overrides
    def setUp(self):
        super().setUp()
        self.plugins_root = self.FIXTURES_ROOT / "plugins"
        self.project_a_fixtures_root = self.plugins_root / "project_a"
        self.project_b_fixtures_root = self.plugins_root / "project_b"
        self.project_c_fixtures_root = self.plugins_root / "project_c"
        self.project_d_fixtures_root = self.plugins_root / "project_d"

    def test_no_plugins(self):
        available_plugins = list(discover_plugins())
        self.assertEqual(0, len(available_plugins))

    def test_namespace_plugins_are_discovered_and_imported(self):
        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b, tempfile.TemporaryDirectory() as temp_dir_c:
            distutils.dir_util.copy_tree(self.project_a_fixtures_root, temp_dir_a)
            distutils.dir_util.copy_tree(self.project_b_fixtures_root, temp_dir_b)
            distutils.dir_util.copy_tree(self.project_c_fixtures_root, temp_dir_c)

            # We make plugins "a" and "c" available as packages, each from other directories, as if they were
            # separate installed projects ("global" usage of the plugins).
            # We move to another directory with a different plugin "b", as if it were another separate project
            # which is not installed ("local" usage of the plugin declared in the namespace).
            with pip_install(temp_dir_a, "a"), pushd(temp_dir_b), push_python_path(
                "."
            ), pip_install(temp_dir_c, "c"):
                # In general when we run scripts or commands in a project, the current directory is the root of it
                # and is part of the path. So we emulate this here with `push_python_path`.

                available_plugins = list(allennlp.common.plugins.discover_plugins())
                self.assertEqual(3, len(available_plugins))

                allennlp.common.plugins.import_plugins()
                # As a secondary effect of importing, the new subcommands should be available.
                subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
                self.assertIn("A", subcommands_available)
                self.assertIn("B", subcommands_available)
                self.assertIn("C", subcommands_available)

    def test_namespace_and_file_plugins_are_discovered_and_imported(self):
        # We make plugins "a" and "c" available as packages, each from other directories, as if they were
        # separate installed projects ("global" usage of the plugins).
        # We move to another directory with a different plugin "b", as if it were another separate project
        # which is not installed ("local" usage of the plugin declared in a file).
        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_c, tempfile.TemporaryDirectory() as temp_dir_d:
            distutils.dir_util.copy_tree(self.project_a_fixtures_root, temp_dir_a)
            distutils.dir_util.copy_tree(self.project_c_fixtures_root, temp_dir_c)
            distutils.dir_util.copy_tree(self.project_d_fixtures_root, temp_dir_d)

            with pip_install(temp_dir_a, "a"), pushd(temp_dir_d), push_python_path(
                "."
            ), pip_install(temp_dir_c, "c"):
                # In general when we run scripts or commands in a project, the current directory is the root of it
                # and is part of the path. So we emulate this here with `push_python_path`.

                available_plugins = list(allennlp.common.plugins.discover_plugins())
                self.assertEqual(3, len(available_plugins))

                allennlp.common.plugins.import_plugins()
                # As a secondary effect of importing, the new subcommands should be available.
                subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
                self.assertIn("A", subcommands_available)
                self.assertIn("C", subcommands_available)
                self.assertIn("D", subcommands_available)

    def test_reload_plugins_with_different_paths(self):
        with pip_install(self.project_a_fixtures_root, "a"):
            available_plugins = list(allennlp.common.plugins.discover_plugins())
            self.assertEqual(1, len(available_plugins))

            allennlp.common.plugins.import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertIn("A", subcommands_available)
            self.assertNotIn("C", subcommands_available)

        with pip_install(self.project_c_fixtures_root, "c"):
            available_plugins = list(allennlp.common.plugins.discover_plugins())
            self.assertEqual(1, len(available_plugins))

            allennlp.common.plugins.import_plugins()
            subcommands_available = {t.__name__ for t in Subcommand.__subclasses__()}
            self.assertNotIn("A", subcommands_available)
            self.assertIn("C", subcommands_available)

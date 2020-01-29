import io
import shutil
import sys
from contextlib import redirect_stdout

import pytest
from overrides import overrides

from allennlp.commands import main
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.plugins import discover_plugins
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import push_python_path
from allennlp.tests.common.plugins_util import pip_install, push_python_project


class TestMain(AllenNlpTestCase):
    def test_fails_on_unknown_command(self):
        sys.argv = [
            "bogus",  # command
            "unknown_model",  # model_name
            "bogus file",  # input_file
            "--output-file",
            "bogus out file",
            "--silent",
        ]

        with self.assertRaises(SystemExit) as cm:
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_subcommand_overrides(self):
        called = False

        def do_nothing(_):
            nonlocal called
            called = True

        @Subcommand.register("evaluate", exist_ok=True)
        class FakeEvaluate(Subcommand):  # noqa
            @overrides
            def add_subparser(self, parser):
                subparser = parser.add_parser(self.name, description="fake", help="fake help")
                subparser.set_defaults(func=do_nothing)
                return subparser

        sys.argv = ["allennlp", "evaluate"]
        main()
        assert called

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / "testpackage"
        packagedir.mkdir()
        (packagedir / "__init__.py").touch()

        # And add that directory to the path
        with push_python_path(self.TEST_DIR):
            # Write out a duplicate model there, but registered under a different name.
            from allennlp.models import simple_tagger

            with open(simple_tagger.__file__) as model_file:
                code = model_file.read().replace(
                    """@Model.register("simple_tagger")""",
                    """@Model.register("duplicate-test-tagger")""",
                )

            with open(packagedir / "model.py", "w") as new_model_file:
                new_model_file.write(code)

            # Copy fixture there too.
            shutil.copy(self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv", self.TEST_DIR)
            data_path = str(self.TEST_DIR / "sequence_tagging.tsv")

            # Write out config file
            config_path = self.TEST_DIR / "config.json"
            config_json = """{
                    "model": {
                            "type": "duplicate-test-tagger",
                            "text_field_embedder": {
                                    "token_embedders": {
                                            "tokens": {
                                                    "type": "embedding",
                                                    "embedding_dim": 5
                                            }
                                    }
                            },
                            "encoder": {
                                    "type": "lstm",
                                    "input_size": 5,
                                    "hidden_size": 7,
                                    "num_layers": 2
                            }
                    },
                    "dataset_reader": {"type": "sequence_tagging"},
                    "train_data_path": "$$$",
                    "validation_data_path": "$$$",
                    "iterator": {"type": "basic", "batch_size": 2},
                    "trainer": {
                            "num_epochs": 2,
                            "optimizer": "adam"
                    }
                }""".replace(
                "$$$", data_path
            )
            with open(config_path, "w") as config_file:
                config_file.write(config_json)

            serialization_dir = self.TEST_DIR / "serialization"

            # Run train with using the non-allennlp module.
            sys.argv = ["allennlp", "train", str(config_path), "-s", str(serialization_dir)]

            # Shouldn't be able to find the model.
            with pytest.raises(ConfigurationError):
                main()

            # Now add the --include-package flag and it should work.
            # We also need to add --recover since the output directory already exists.
            sys.argv.extend(["--recover", "--include-package", "testpackage"])

            main()

            # Rewrite out config file, but change a value.
            with open(config_path, "w") as new_config_file:
                new_config_file.write(config_json.replace('"num_epochs": 2,', '"num_epochs": 4,'))

            # This should fail because the config.json does not match that in the serialization directory.
            with pytest.raises(ConfigurationError):
                main()

    def test_file_plugin_loaded(self):
        plugins_root = self.FIXTURES_ROOT / "plugins"
        # "d" sets a "local" file plugin, because it's supposed to be run from that directory
        # and has a ".allennlp_plugins" file in it.
        project_d_fixtures_root = plugins_root / "project_d"

        sys.argv = ["allennlp"]

        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

        with push_python_project(project_d_fixtures_root):
            main()
            subcommands_available = Subcommand.list_available()
            self.assertIn("d", subcommands_available)

    def test_namespace_plugin_loaded(self):
        plugins_root = self.FIXTURES_ROOT / "plugins"
        # "a" sets a "global" namespace plugin, because it's gonna be installed with pip.
        project_a_fixtures_root = plugins_root / "project_a"

        sys.argv = ["allennlp"]

        available_plugins = set(discover_plugins())
        self.assertSetEqual(set(), available_plugins)

        with pip_install(project_a_fixtures_root, "a"):
            main()

        subcommands_available = Subcommand.list_available()
        self.assertIn("a", subcommands_available)

    def test_subcommand_plugin_is_available(self):
        plugins_root = self.FIXTURES_ROOT / "plugins"
        allennlp_server_fixtures_root = plugins_root / "allennlp_server"

        sys.argv = ["allennlp"]

        with pip_install(
            allennlp_server_fixtures_root, "allennlp_server"
        ), io.StringIO() as buf, redirect_stdout(buf):
            main()
            output = buf.getvalue()

        self.assertIn("    serve", output)

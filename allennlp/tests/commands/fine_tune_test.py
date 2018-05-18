# pylint: disable=invalid-name,no-self-use
import argparse
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.fine_tune import FineTune, fine_tune_model_from_file_paths, fine_tune_model_from_args

class TestFineTune(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        self.config_file = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        self.serialization_dir = str(self.TEST_DIR / 'fine_tune')

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title='Commands', metavar='')
        FineTune().add_subparser('fine-tune', subparsers)


    def test_fine_tune_model_runs_from_file_paths(self):
        initial_working_dir = os.getcwd()
        # Change directory to module root.
        os.chdir(self.MODULE_ROOT)

        fine_tune_model_from_file_paths(model_archive_path=self.model_archive,
                                        config_file=self.config_file,
                                        serialization_dir=self.serialization_dir)
        # Change directory back to what it was initially
        os.chdir(initial_working_dir)

    def test_fine_tune_runs_from_parser_arguments(self):
        initial_working_dir = os.getcwd()
        # Change directory to module root.
        os.chdir(self.MODULE_ROOT)

        raw_args = ["fine-tune",
                    "-m", self.model_archive,
                    "-c", self.config_file,
                    "-s", self.serialization_dir]

        args = self.parser.parse_args(raw_args)

        assert args.func == fine_tune_model_from_args
        assert args.model_archive == self.model_archive
        assert args.config_file == self.config_file
        assert args.serialization_dir == self.serialization_dir
        fine_tune_model_from_args(args)
        # Change directory back to what it was initially
        os.chdir(initial_working_dir)

    def test_fine_tune_fails_without_required_args(self):
        # Configuration file is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args(["fine-tune", "-m", "path/to/archive", "-s", "serialization_dir"])
            assert context.exception.code == 2  # argparse code for incorrect usage

        # Serialization dir is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args(["fine-tune", "-m", "path/to/archive", "-c", "path/to/config"])
            assert context.exception.code == 2  # argparse code for incorrect usage

        # Model archive is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args(["fine-tune", "-s", "serialization_dir", "-c", "path/to/config"])
            assert context.exception.code == 2  # argparse code for incorrect usage

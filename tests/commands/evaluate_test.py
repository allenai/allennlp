# pylint: disable=invalid-name,no-self-use
import argparse
import json
import os
import shutil
import sys
import tarfile

from flaky import flaky
import pytest

from allennlp.commands.evaluate import evaluate_from_args, Evaluate
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import archive_model


class TestEvaluate(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title='Commands', metavar='')
        Evaluate().add_subparser('evaluate', subparsers)


    @flaky
    def test_evaluate_from_args(self):
        snake_args = ["evaluate", "tests/fixtures/bidaf/serialization/model.tar.gz",
                      "--evaluation_data_file", "tests/fixtures/data/squad.json",
                      "--cuda_device", "-1"]

        kebab_args = ["evaluate", "tests/fixtures/bidaf/serialization/model.tar.gz",
                      "--evaluation-data-file", "tests/fixtures/data/squad.json",
                      "--cuda-device", "-1"]

        for raw_args in [snake_args, kebab_args]:
            args = self.parser.parse_args(raw_args)
            metrics = evaluate_from_args(args)
            assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1'}

    def test_external_modules(self):
        sys.path.insert(0, self.TEST_DIR)

        original_serialization_dir = 'tests/fixtures/bidaf/serialization'
        serialization_dir = os.path.join(self.TEST_DIR, 'serialization')
        shutil.copytree(original_serialization_dir, serialization_dir)

        # Get original model config
        tf = tarfile.open(os.path.join(original_serialization_dir, 'model.tar.gz'))
        tf.extract('config.json', self.TEST_DIR)

        # Write out modified config file
        params = Params.from_file(os.path.join(self.TEST_DIR, 'config.json'))
        params['model']['type'] = 'bidaf-duplicate'

        config_file = os.path.join(serialization_dir, 'model_params.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(params.as_dict(quiet=True)))

        # And create an archive
        archive_model(serialization_dir)

        # Write out modified model.py
        module_dir = os.path.join(self.TEST_DIR, 'bidaf_duplicate')
        os.makedirs(module_dir)

        from allennlp.models.reading_comprehension import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace("""@Model.register("bidaf")""",
                                    """@Model.register('bidaf-duplicate')""")

        with open(os.path.join(module_dir, 'model.py'), 'w') as f:
            f.write(code)

        archive_file = os.path.join(serialization_dir, 'model.tar.gz')

        raw_args = ["evaluate", archive_file,
                    "--evaluation-data-file", "tests/fixtures/data/squad.json"]

        args = self.parser.parse_args(raw_args)

        # Raise configuration error without extra modules
        with pytest.raises(ConfigurationError):
            metrics = evaluate_from_args(args)

        # Specify the additional module
        raw_args.extend(['--include-package', 'bidaf_duplicate'])
        args = self.parser.parse_args(raw_args)
        metrics = evaluate_from_args(args)

        assert metrics.keys() == {'span_acc', 'end_acc', 'start_acc', 'em', 'f1'}

        sys.path.remove(self.TEST_DIR)

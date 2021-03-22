import argparse
import sys

from allennlp.commands import main
from allennlp.commands.checklist import CheckList
from allennlp.common.testing import AllenNlpTestCase


class TestCheckList(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.archive_file = (
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        self.task_suite = "sentiment-analysis-vocabulary"

    def test_add_checklist_subparser(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        CheckList().add_subparser(subparsers)

        kebab_args = [
            "checklist",  # command
            "/path/to/archive",  # archive
            "task-suite-name-or-path",  # task suite
            "--output-file",
            "/dev/null",
            "--cuda-device",
            "0",
            "--silent",
        ]

        args = parser.parse_args(kebab_args)

        assert args.func.__name__ == "_run_suite"
        assert args.archive_file == "/path/to/archive"
        assert args.task_suite == "task-suite-name-or-path"
        assert args.output_file == "/dev/null"
        assert args.cuda_device == 0
        assert args.silent

    def test_works_with_known_model(self):

        sys.argv = [
            "__main__.py",  # executable
            "checklist",  # command
            str(self.archive_file),
            str(self.task_suite),
            "--task-suite-args",
            '{"positive": 1, "negative": 0, "neutral": null}',
        ]

        main()

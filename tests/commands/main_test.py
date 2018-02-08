from unittest import TestCase
import logging
import sys

from allennlp.commands import main
from allennlp.commands.subcommand import Subcommand

class TestMain(TestCase):

    def test_fails_on_unknown_command(self):
        sys.argv = ["bogus",         # command
                    "unknown_model", # model_name
                    "bogus file",    # input_file
                    "--output-file", "bogus out file",
                    "--silent"]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_warn_on_deprecated_flags(self):
        sys.argv = ["[executable]",
                    "evaluate", "tests/fixtures/bidaf/serialization/model.tar.gz",
                    "--evaluation_data_file", "tests/fixtures/data/squad.json",
                    "--cuda_device", "-1"]


        with self.assertLogs(level=logging.WARNING) as context:
            main()
            assert set(context.output) == {
                    'WARNING:allennlp.commands:Argument name --evaluation_data_file is deprecated '
                    '(and will likely go away at some point), please use --evaluation-data-file instead',

                    'WARNING:allennlp.commands:Argument name --cuda_device is deprecated '
                    '(and will likely go away at some point), please use --cuda-device instead',
            }


    def test_subcommand_overrides(self):
        def do_nothing(_):
            pass

        class FakeEvaluate(Subcommand):
            add_subparser_called = False

            def add_subparser(self, name, parser):
                subparser = parser.add_parser(name,
                                              description="fake",
                                              help="fake help")

                subparser.set_defaults(func=do_nothing)
                self.add_subparser_called = True

                return subparser

        fake_evaluate = FakeEvaluate()

        sys.argv = ["evaluate"]
        main(subcommand_overrides={"evaluate": fake_evaluate})

        assert fake_evaluate.add_subparser_called

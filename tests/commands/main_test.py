from unittest import TestCase

from allennlp.commands.main import main

class TestMain(TestCase):

    def test_fails_on_unknown_command(self):
        args = ["bogus",         # command
                "unknown_model", # model_name
                "bogus file",    # input_file
                "--output-file", "bogus out file",
                "--silent"]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main(args)

        assert cm.exception.code == 2  # argparse code for incorrect usage

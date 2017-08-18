# pylint: disable=no-self-use,invalid-name
import argparse
import json
import os
import tempfile
from unittest import TestCase

from allennlp.commands.main import main
from allennlp.commands.predict import add_subparser, predict


class TestPredict(TestCase):

    def test_add_predict_subparser(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers)

        raw_args = ["predict",     # command
                    "/path/to/model", # model
                    "/path/to/data",   # input_file
                    "--output-file", "outfile",
                    "--print"]

        args = parser.parse_args(raw_args)

        assert args.func == predict
        assert args.config_file == "/path/to/model"
        assert args.input_file == "/path/to/data"
        assert args.output_file == "outfile"
        assert args.print

    def test_works_with_known_model(self):
        tempdir = tempfile.mkdtemp()
        infile = os.path.join(tempdir, "inputs.txt")
        outfile = os.path.join(tempdir, "outputs.txt")

        with open(infile, 'w') as f:
            f.write("""{"sentence": "this is a great sentence"}\n""")
            f.write("""{"sentence": "this is a less great sentence"}\n""")

        args = ["predict",     # command
                "tests/fixtures/srl/experiment.json",
                infile,     # input_file
                "--output-file", outfile,
                "--print"]

        main(args)

        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2


    def test_fails_without_required_args(self):
        args = ["predict",          # command
                "/path/to/model",   # model, but no input file
               ]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main(args)

        assert cm.exception.code == 2  # argparse code for incorrect usage

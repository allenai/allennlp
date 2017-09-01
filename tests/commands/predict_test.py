# pylint: disable=no-self-use,invalid-name
import argparse
import json
import os
import sys
import tempfile
from unittest import TestCase

from allennlp.commands import main
from allennlp.commands.predict import add_subparser, predict


class TestPredict(TestCase):

    def test_add_predict_subparser(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers)

        raw_args = ["predict",          # command
                    "/path/to/archive", # archive
                    "/dev/null",    # input_file
                    "--output-file", "/dev/null",
                    "--silent"]

        args = parser.parse_args(raw_args)

        assert args.func == predict
        assert args.archive_file == "/path/to/archive"
        assert args.silent

    def test_works_with_known_model(self):
        tempdir = tempfile.mkdtemp()
        infile = os.path.join(tempdir, "inputs.txt")
        outfile = os.path.join(tempdir, "outputs.txt")

        with open(infile, 'w') as f:
            f.write("""{"passage": "the seahawks won the super bowl in 2016", """
                    """ "question": "when did the seahawks win the super bowl?"}\n""")
            f.write("""{"passage": "the mariners won the super bowl in 2037", """
                    """ "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    "tests/fixtures/bidaf/serialization/model.tar.gz",
                    infile,     # input_file
                    "--output-file", outfile,
                    "--silent"]

        main()

        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"span_start_probs", "span_end_probs", "best_span",
                                          "best_span_str", "tokens"}

    def test_fails_without_required_args(self):
        sys.argv = ["run.py",            # executable
                    "predict",           # command
                    "/path/to/archive",  # archive, but no input file
                   ]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

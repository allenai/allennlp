# pylint: disable=no-self-use,invalid-name

import json
import os
import tempfile
from unittest import TestCase

from allennlp.__main__ import main


class TestMain(TestCase):

    def test_works_with_known_model(self):
        tempdir = tempfile.mkdtemp()
        infile = os.path.join(tempdir, "inputs.txt")
        outfile = os.path.join(tempdir, "outputs.txt")

        with open(infile, 'w') as f:
            f.write("""{"input": "forward"}\n""")
            f.write("""{"input": "drawkcab"}\n""")

        args = ["bulk",     # command
                "reverser", # model_name
                infile,     # input_file
                "--output-file", outfile,
                "--print"]

        main(args)

        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2
        assert lines[0] == {"model_name": "reverser", "input": "forward", "output": "drawrof"}
        assert lines[1] == {"model_name": "reverser", "input": "drawkcab", "output": "backward"}

    def test_fails_without_required_args(self):
        args = ["bulk",          # command
                "reverser",      # model_name, but no input file
               ]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main(args)

        assert cm.exception.code == -1


    def test_fails_on_unknown_model(self):
        args = ["bulk",          # command
                "unknown_model", # model_name
                "bogus file",    # input_file
                "--output-file", "bogus out file",
                "--print"]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main(args)

        assert cm.exception.code == -1

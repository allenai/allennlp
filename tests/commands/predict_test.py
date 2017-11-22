# pylint: disable=no-self-use,invalid-name
import argparse
import json
import os
import sys
import tempfile
from unittest import TestCase

from allennlp.common.util import JsonDict
from allennlp.commands import main
from allennlp.commands.predict import Predict, DEFAULT_PREDICTORS
from allennlp.service.predictors import Predictor, BidafPredictor


class TestPredict(TestCase):

    def test_add_predict_subparser(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Predict(DEFAULT_PREDICTORS).add_subparser('predict', subparsers)

        snake_args = ["predict",          # command
                      "/path/to/archive", # archive
                      "/dev/null",        # input_file
                      "--output-file", "/dev/null",  # this one was always kebab-case
                      "--batch_size", "10",
                      "--cuda_device", "0",
                      "--silent"]

        kebab_args = ["predict",          # command
                      "/path/to/archive", # archive
                      "/dev/null",        # input_file
                      "--output-file", "/dev/null",
                      "--batch-size", "10",
                      "--cuda-device", "0",
                      "--silent"]

        for raw_args in [snake_args, kebab_args]:
            args = parser.parse_args(raw_args)

            assert args.func.__name__ == 'predict_inner'
            assert args.archive_file == "/path/to/archive"
            assert args.output_file.name == "/dev/null"
            assert args.batch_size == 10
            assert args.cuda_device == 0
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
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "span_start_probs", "span_end_probs", "best_span",
                                          "best_span_str"}

    def test_batch_prediction_works_with_known_model(self):
        tempdir = tempfile.mkdtemp()
        infile = os.path.join(tempdir, "inputs.txt")
        outfile = os.path.join(tempdir, "outputs.txt")

        with open(infile, 'w') as f:
            f.write("""{"passage": "the seahawks won the super bowl in 2016", """
                    """ "question": "when did the seahawks win the super bowl?"}\n""")
            f.write("""{"passage": "the mariners won the super bowl in 2037", """
                    """ "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = ["run.py",  # executable
                    "predict",  # command
                    "tests/fixtures/bidaf/serialization/model.tar.gz",
                    infile,  # input_file
                    "--output-file", outfile,
                    "--silent",
                    "--batch_size", '2']

        main()

        assert os.path.exists(outfile)
        with open(outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "span_start_probs", "span_end_probs", "best_span",
                                          "best_span_str"}

    def test_fails_without_required_args(self):
        sys.argv = ["run.py",            # executable
                    "predict",           # command
                    "/path/to/archive",  # archive, but no input file
                   ]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_can_override_predictors(self):

        @Predictor.register('bidaf-override')  # pylint: disable=unused-variable
        class Bidaf2Predictor(BidafPredictor):
            """same as bidaf predictor but with an extra field"""
            def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
                result = super().predict_json(inputs)
                result["overridden"] = True
                return result

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

        main(predictor_overrides={'bidaf': 'bidaf-override'})
        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "span_start_probs", "span_end_probs", "best_span",
                                          "best_span_str", "overridden"}

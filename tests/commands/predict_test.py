# pylint: disable=no-self-use,invalid-name
import argparse
import csv
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands import main
from allennlp.commands.predict import Predict, DEFAULT_PREDICTORS
from allennlp.service.predictors import Predictor, BidafPredictor


class TestPredict(AllenNlpTestCase):

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
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str"}

        shutil.rmtree(tempdir)

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
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str"}

        shutil.rmtree(tempdir)

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
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str", "overridden"}

        shutil.rmtree(tempdir)

    def test_can_specify_predictor(self):

        @Predictor.register('bidaf-explicit')  # pylint: disable=unused-variable
        class Bidaf3Predictor(BidafPredictor):
            """same as bidaf predictor but with an extra field"""
            def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
                result = super().predict_json(inputs)
                result["explicit"] = True
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
                    "--predictor", "bidaf-explicit",
                    "--silent"]

        main()
        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str", "explicit"}

        shutil.rmtree(tempdir)

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = os.path.join(self.TEST_DIR, 'testpackage')
        pathlib.Path(packagedir).mkdir()
        pathlib.Path(os.path.join(packagedir, '__init__.py')).touch()

        # And add that directory to the path
        sys.path.insert(0, self.TEST_DIR)

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.service.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace("""@Predictor.register('machine-comprehension')""",
                                    """@Predictor.register('duplicate-test-predictor')""")

        with open(os.path.join(packagedir, 'predictor.py'), 'w') as f:
            f.write(code)

        infile = os.path.join(self.TEST_DIR, "inputs.txt")
        outfile = os.path.join(self.TEST_DIR, "outputs.txt")

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
                    "--predictor", "duplicate-test-predictor",
                    "--silent"]

        # Should raise ConfigurationError, because predictor is unknown
        with pytest.raises(ConfigurationError):
            main()

        # But once we include testpackage, it should be known
        sys.argv.extend(["--include-package", "testpackage"])
        main()

        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str"}

        sys.path.remove(self.TEST_DIR)

    def test_alternative_file_formats(self):
        tempdir = tempfile.mkdtemp()
        infile = os.path.join(tempdir, "inputs.txt")
        outfile = os.path.join(tempdir, "outputs.txt")

        @Predictor.register('bidaf-csv')  # pylint: disable=unused-variable
        class BidafCsvPredictor(BidafPredictor):
            """same as bidaf predictor but using CSV inputs and outputs"""
            def load_line(self, line: str) -> JsonDict:
                reader = csv.reader([line])
                passage, question = next(reader)
                return {"passage": passage, "question": question}

            def dump_line(self, outputs: JsonDict) -> str:
                output = io.StringIO()
                writer = csv.writer(output)
                row = [outputs["span_start_probs"][0],
                       outputs["span_end_probs"][0],
                       *outputs["best_span"],
                       outputs["best_span_str"]]

                writer.writerow(row)
                return output.getvalue()

        with open(infile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["the seahawks won the super bowl in 2016",
                             "when did the seahawks win the super bowl?"])
            writer.writerow(["the mariners won the super bowl in 2037",
                             "when did the mariners win the super bowl?"])


        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    "tests/fixtures/bidaf/serialization/model.tar.gz",
                    infile,     # input_file
                    "--output-file", outfile,
                    "--silent"]

        main(predictor_overrides={'bidaf': 'bidaf-csv'})
        assert os.path.exists(outfile)

        with open(outfile, 'r') as f:
            reader = csv.reader(f)
            results = [row for row in reader]

        assert len(results) == 2
        for row in results:
            assert len(row) == 5
            start_prob, end_prob, span_start, span_end, span = row
            for prob in (start_prob, end_prob):
                assert 0 <= float(prob) <= 1
            assert 0 <= int(span_start) <= int(span_end) <= 8
            assert span != ''

        shutil.rmtree(tempdir)

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
from allennlp.commands.predict import Predict
from allennlp.predictors import Predictor, BidafPredictor


class TestPredict(AllenNlpTestCase):
    def setUp(self):
        super(TestPredict, self).setUp()
        self.bidaf_model_path = (self.FIXTURES_ROOT / "bidaf" /
                                 "serialization" / "model.tar.gz")
        self.bidaf_data_path = self.FIXTURES_ROOT / 'data' / 'squad.json'
        self.atis_model_path = (self.FIXTURES_ROOT / "semantic_parsing" / "atis" /
                                "serialization" / "model.tar.gz")
        self.atis_data_path = self.FIXTURES_ROOT / 'data' / 'atis' / 'sample.json'
        self.tempdir = pathlib.Path(tempfile.mkdtemp())
        self.infile = self.tempdir / "inputs.txt"
        self.outfile = self.tempdir / "outputs.txt"

    def test_add_predict_subparser(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Predict().add_subparser('predict', subparsers)

        kebab_args = ["predict",          # command
                      "/path/to/archive", # archive
                      "/dev/null",        # input_file
                      "--output-file", "/dev/null",
                      "--batch-size", "10",
                      "--cuda-device", "0",
                      "--silent"]

        args = parser.parse_args(kebab_args)

        assert args.func.__name__ == '_predict'
        assert args.archive_file == "/path/to/archive"
        assert args.output_file == "/dev/null"
        assert args.batch_size == 10
        assert args.cuda_device == 0
        assert args.silent

    def test_works_with_known_model(self):
        with open(self.infile, 'w') as f:
            f.write("""{"passage": "the seahawks won the super bowl in 2016", """
                    """ "question": "when did the seahawks win the super bowl?"}\n""")
            f.write("""{"passage": "the mariners won the super bowl in 2037", """
                    """ "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.bidaf_model_path),
                    str(self.infile),     # input_file
                    "--output-file", str(self.outfile),
                    "--silent"]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str"}

        shutil.rmtree(self.tempdir)

    def test_using_dataset_reader_works_with_known_model(self):

        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.bidaf_model_path),
                    str(self.bidaf_data_path),     # input_file
                    "--output-file", str(self.outfile),
                    "--silent",
                    "--use-dataset-reader"]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 5
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str", "loss"}

        shutil.rmtree(self.tempdir)

    def test_uses_correct_dataset_reader(self):
        # pylint: disable=protected-access
        # The ATIS archive has both a training and validation ``DatasetReader``
        # with different values for ``keep_if_unparseable`` (``True`` for validation
        # and ``False`` for training). We create a new ``Predictor`` class that
        # outputs this value so we can test which ``DatasetReader`` was used.
        @Predictor.register('test-predictor')
        class _TestPredictor(Predictor):
            # pylint: disable=abstract-method
            def dump_line(self, outputs: JsonDict) -> str:
                data = {'keep_if_unparseable': self._dataset_reader._keep_if_unparseable}  # type: ignore
                return json.dumps(data) + '\n'

        # --use-dataset-reader argument only should use validation
        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.atis_model_path),
                    str(self.atis_data_path),     # input_file
                    "--output-file", str(self.outfile),
                    "--silent",
                    "--predictor", "test-predictor",
                    "--use-dataset-reader"]
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]
            assert results[0]['keep_if_unparseable'] is True

        # --use-dataset-reader, override with train
        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.atis_model_path),
                    str(self.atis_data_path),     # input_file
                    "--output-file", str(self.outfile),
                    "--silent",
                    "--predictor", "test-predictor",
                    "--use-dataset-reader",
                    "--dataset-reader-choice", "train"]
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]
            assert results[0]['keep_if_unparseable'] is False

        # --use-dataset-reader, override with train
        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.atis_model_path),
                    str(self.atis_data_path),     # input_file
                    "--output-file", str(self.outfile),
                    "--silent",
                    "--predictor", "test-predictor",
                    "--use-dataset-reader",
                    "--dataset-reader-choice", "validation"]
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]
            assert results[0]['keep_if_unparseable'] is True

        # No --use-dataset-reader flag, fails because the loading logic
        # is not implemented in the testing predictor
        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.atis_model_path),
                    str(self.atis_data_path),     # input_file
                    "--output-file", str(self.outfile),
                    "--silent",
                    "--predictor", "test-predictor"]
        with self.assertRaises(NotImplementedError):
            main()

    def test_batch_prediction_works_with_known_model(self):
        with open(self.infile, 'w') as f:
            f.write("""{"passage": "the seahawks won the super bowl in 2016", """
                    """ "question": "when did the seahawks win the super bowl?"}\n""")
            f.write("""{"passage": "the mariners won the super bowl in 2037", """
                    """ "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = ["run.py",  # executable
                    "predict",  # command
                    str(self.bidaf_model_path),
                    str(self.infile),  # input_file
                    "--output-file", str(self.outfile),
                    "--silent",
                    "--batch-size", '2']

        main()

        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str"}

        shutil.rmtree(self.tempdir)

    def test_fails_without_required_args(self):
        sys.argv = ["run.py",            # executable
                    "predict",           # command
                    "/path/to/archive",  # archive, but no input file
                   ]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_can_specify_predictor(self):

        @Predictor.register('bidaf-explicit')  # pylint: disable=unused-variable
        class Bidaf3Predictor(BidafPredictor):
            """same as bidaf predictor but with an extra field"""
            def predict_json(self, inputs: JsonDict) -> JsonDict:
                result = super().predict_json(inputs)
                result["explicit"] = True
                return result

        with open(self.infile, 'w') as f:
            f.write("""{"passage": "the seahawks won the super bowl in 2016", """
                    """ "question": "when did the seahawks win the super bowl?"}\n""")
            f.write("""{"passage": "the mariners won the super bowl in 2037", """
                    """ "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.bidaf_model_path),
                    str(self.infile),     # input_file
                    "--output-file", str(self.outfile),
                    "--predictor", "bidaf-explicit",
                    "--silent"]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str", "explicit"}

        shutil.rmtree(self.tempdir)

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / 'testpackage'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / '__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, str(self.TEST_DIR))

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace("""@Predictor.register('machine-comprehension')""",
                                    """@Predictor.register('duplicate-test-predictor')""")

        with open(os.path.join(packagedir, 'predictor.py'), 'w') as f:
            f.write(code)

        self.infile = os.path.join(self.TEST_DIR, "inputs.txt")
        self.outfile = os.path.join(self.TEST_DIR, "outputs.txt")

        with open(self.infile, 'w') as f:
            f.write("""{"passage": "the seahawks won the super bowl in 2016", """
                    """ "question": "when did the seahawks win the super bowl?"}\n""")
            f.write("""{"passage": "the mariners won the super bowl in 2037", """
                    """ "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.bidaf_model_path),
                    str(self.infile),     # input_file
                    "--output-file", str(self.outfile),
                    "--predictor", "duplicate-test-predictor",
                    "--silent"]

        # Should raise ConfigurationError, because predictor is unknown
        with pytest.raises(ConfigurationError):
            main()

        # But once we include testpackage, it should be known
        sys.argv.extend(["--include-package", "testpackage"])
        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, 'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == {"span_start_logits", "span_end_logits",
                                          "passage_question_attention", "question_tokens",
                                          "passage_tokens", "span_start_probs", "span_end_probs",
                                          "best_span", "best_span_str"}

        sys.path.remove(str(self.TEST_DIR))

    def test_alternative_file_formats(self):
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

        with open(self.infile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["the seahawks won the super bowl in 2016",
                             "when did the seahawks win the super bowl?"])
            writer.writerow(["the mariners won the super bowl in 2037",
                             "when did the mariners win the super bowl?"])

        sys.argv = ["run.py",      # executable
                    "predict",     # command
                    str(self.bidaf_model_path),
                    str(self.infile),     # input_file
                    "--output-file", str(self.outfile),
                    "--predictor", 'bidaf-csv',
                    "--silent"]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile, 'r') as f:
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

        shutil.rmtree(self.tempdir)

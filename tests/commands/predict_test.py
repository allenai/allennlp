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

from allennlp.commands import main
from allennlp.commands.predict import Predict
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import JsonDict, push_python_path
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor, TextClassifierPredictor


class TestPredict(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.classifier_model_path = (
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        self.classifier_data_path = (
            self.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_corpus.jsonl"
        )
        self.tempdir = pathlib.Path(tempfile.mkdtemp())
        self.infile = self.tempdir / "inputs.txt"
        self.outfile = self.tempdir / "outputs.txt"

    def test_add_predict_subparser(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        Predict().add_subparser(subparsers)

        kebab_args = [
            "predict",  # command
            "/path/to/archive",  # archive
            "/dev/null",  # input_file
            "--output-file",
            "/dev/null",
            "--batch-size",
            "10",
            "--cuda-device",
            "0",
            "--silent",
        ]

        args = parser.parse_args(kebab_args)

        assert args.func.__name__ == "_predict"
        assert args.archive_file == "/path/to/archive"
        assert args.output_file == "/dev/null"
        assert args.batch_size == 10
        assert args.cuda_device == 0
        assert args.silent

    def test_works_with_known_model(self):
        with open(self.infile, "w") as f:
            f.write("""{"sentence": "the seahawks won the super bowl in 2016"}\n""")
            f.write("""{"sentence": "the mariners won the super bowl in 2037"}\n""")

        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.infile),  # input_file
            "--output-file",
            str(self.outfile),
            "--silent",
        ]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"label", "logits", "probs", "tokens", "token_ids"}

        shutil.rmtree(self.tempdir)

    def test_using_dataset_reader_works_with_known_model(self):

        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.classifier_data_path),  # input_file
            "--output-file",
            str(self.outfile),
            "--silent",
            "--use-dataset-reader",
        ]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 3
        for result in results:
            assert set(result.keys()) == {"label", "logits", "loss", "probs", "tokens", "token_ids"}

        shutil.rmtree(self.tempdir)

    def test_uses_correct_dataset_reader(self):
        # We're going to use a fake predictor for this test, just checking that we loaded the
        # correct dataset reader.  We'll also create a fake dataset reader that subclasses the
        # expected one, and specify that one for validation.
        @Predictor.register("test-predictor")
        class _TestPredictor(Predictor):
            def dump_line(self, outputs: JsonDict) -> str:
                data = {"dataset_reader_type": type(self._dataset_reader).__name__}  # type: ignore
                return json.dumps(data) + "\n"

            def load_line(self, line: str) -> JsonDict:
                raise NotImplementedError

        @DatasetReader.register("fake-reader")
        class FakeDatasetReader(TextClassificationJsonReader):
            pass

        # --use-dataset-reader argument only should use validation
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.classifier_data_path),  # input_file
            "--output-file",
            str(self.outfile),
            "--overrides",
            '{"validation_dataset_reader": {"type": "fake-reader"}}',
            "--silent",
            "--predictor",
            "test-predictor",
            "--use-dataset-reader",
        ]
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]
            assert results[0]["dataset_reader_type"] == "FakeDatasetReader"

        # --use-dataset-reader, override with train
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.classifier_data_path),  # input_file
            "--output-file",
            str(self.outfile),
            "--overrides",
            '{"validation_dataset_reader": {"type": "fake-reader"}}',
            "--silent",
            "--predictor",
            "test-predictor",
            "--use-dataset-reader",
            "--dataset-reader-choice",
            "train",
        ]
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]
            assert results[0]["dataset_reader_type"] == "TextClassificationJsonReader"

        # --use-dataset-reader, override with validation
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.classifier_data_path),  # input_file
            "--output-file",
            str(self.outfile),
            "--overrides",
            '{"validation_dataset_reader": {"type": "fake-reader"}}',
            "--silent",
            "--predictor",
            "test-predictor",
            "--use-dataset-reader",
            "--dataset-reader-choice",
            "validation",
        ]
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]
            assert results[0]["dataset_reader_type"] == "FakeDatasetReader"

        # No --use-dataset-reader flag, fails because the loading logic
        # is not implemented in the testing predictor
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.classifier_data_path),  # input_file
            "--output-file",
            str(self.outfile),
            "--overrides",
            '{"validation_dataset_reader": {"type": "fake-reader"}}',
            "--silent",
            "--predictor",
            "test-predictor",
        ]
        with pytest.raises(NotImplementedError):
            main()

    def test_base_predictor(self):
        # Tests when no Predictor is found and the base class implementation is used
        model_path = str(self.classifier_model_path)
        archive = load_archive(model_path)
        model_type = archive.config.get("model").get("type")
        # Makes sure that we don't have a default_predictor for it. Otherwise the base class
        # implementation wouldn't be used
        from allennlp.models import Model

        model_class, _ = Model.resolve_class_name(model_type)
        saved_default_predictor = model_class.default_predictor
        model_class.default_predictor = None
        try:
            # Doesn't use a --predictor
            sys.argv = [
                "__main__.py",  # executable
                "predict",  # command
                model_path,
                str(self.classifier_data_path),  # input_file
                "--output-file",
                str(self.outfile),
                "--silent",
                "--use-dataset-reader",
            ]
            main()
            assert os.path.exists(self.outfile)
            with open(self.outfile, "r") as f:
                results = [json.loads(line) for line in f]

            assert len(results) == 3
            for result in results:
                assert set(result.keys()) == {
                    "logits",
                    "probs",
                    "label",
                    "loss",
                    "tokens",
                    "token_ids",
                }
        finally:
            model_class.default_predictor = saved_default_predictor

    def test_batch_prediction_works_with_known_model(self):
        with open(self.infile, "w") as f:
            f.write("""{"sentence": "the seahawks won the super bowl in 2016"}\n""")
            f.write("""{"sentence": "the mariners won the super bowl in 2037"}\n""")

        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.infile),  # input_file
            "--output-file",
            str(self.outfile),
            "--silent",
            "--batch-size",
            "2",
        ]

        main()

        assert os.path.exists(self.outfile)
        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"label", "logits", "probs", "tokens", "token_ids"}

        shutil.rmtree(self.tempdir)

    def test_fails_without_required_args(self):
        sys.argv = [
            "__main__.py",
            "predict",
            "/path/to/archive",
        ]  # executable  # command  # archive, but no input file

        with pytest.raises(SystemExit) as cm:
            main()

        assert cm.value.code == 2  # argparse code for incorrect usage

    def test_can_specify_predictor(self):
        @Predictor.register("classification-explicit")
        class ExplicitPredictor(TextClassifierPredictor):
            """same as classifier predictor but with an extra field"""

            def predict_json(self, inputs: JsonDict) -> JsonDict:
                result = super().predict_json(inputs)
                result["explicit"] = True
                return result

        with open(self.infile, "w") as f:
            f.write("""{"sentence": "the seahawks won the super bowl in 2016"}\n""")
            f.write("""{"sentence": "the mariners won the super bowl in 2037"}\n""")

        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.infile),  # input_file
            "--output-file",
            str(self.outfile),
            "--predictor",
            "classification-explicit",
            "--silent",
        ]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == {
                "label",
                "logits",
                "explicit",
                "probs",
                "tokens",
                "token_ids",
            }

        shutil.rmtree(self.tempdir)

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / "testpackage"
        packagedir.mkdir()
        (packagedir / "__init__.py").touch()

        # And add that directory to the path
        with push_python_path(self.TEST_DIR):
            # Write out a duplicate predictor there, but registered under a different name.
            from allennlp.predictors import text_classifier

            with open(text_classifier.__file__) as f:
                code = f.read().replace(
                    """@Predictor.register("text_classifier")""",
                    """@Predictor.register("duplicate-test-predictor")""",
                )

            with open(os.path.join(packagedir, "predictor.py"), "w") as f:
                f.write(code)

            self.infile = os.path.join(self.TEST_DIR, "inputs.txt")
            self.outfile = os.path.join(self.TEST_DIR, "outputs.txt")

            with open(self.infile, "w") as f:
                f.write("""{"sentence": "the seahawks won the super bowl in 2016"}\n""")
                f.write("""{"sentence": "the mariners won the super bowl in 2037"}\n""")

            sys.argv = [
                "__main__.py",  # executable
                "predict",  # command
                str(self.classifier_model_path),
                str(self.infile),  # input_file
                "--output-file",
                str(self.outfile),
                "--predictor",
                "duplicate-test-predictor",
                "--silent",
            ]

            # Should raise ConfigurationError, because predictor is unknown
            with pytest.raises(ConfigurationError):
                main()

            # But once we include testpackage, it should be known
            sys.argv.extend(["--include-package", "testpackage"])
            main()

            assert os.path.exists(self.outfile)

            with open(self.outfile, "r") as f:
                results = [json.loads(line) for line in f]

            assert len(results) == 2
            # Overridden predictor should output extra field
            for result in results:
                assert set(result.keys()) == {"label", "logits", "probs", "tokens", "token_ids"}

    def test_alternative_file_formats(self):
        @Predictor.register("classification-csv")
        class CsvPredictor(TextClassifierPredictor):
            """same as classification predictor but using CSV inputs and outputs"""

            def load_line(self, line: str) -> JsonDict:
                reader = csv.reader([line])
                sentence, label = next(reader)
                return {"sentence": sentence, "label": label}

            def dump_line(self, outputs: JsonDict) -> str:
                output = io.StringIO()
                writer = csv.writer(output)
                row = [outputs["label"], *outputs["probs"]]

                writer.writerow(row)
                return output.getvalue()

        with open(self.infile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["the seahawks won the super bowl in 2016", "pos"])
            writer.writerow(["the mariners won the super bowl in 2037", "neg"])

        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.infile),  # input_file
            "--output-file",
            str(self.outfile),
            "--predictor",
            "classification-csv",
            "--silent",
        ]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile) as f:
            reader = csv.reader(f)
            results = [row for row in reader]

        assert len(results) == 2
        for row in results:
            assert len(row) == 3  # label and 2 class probabilities
            label, *probs = row
            for prob in probs:
                assert 0 <= float(prob) <= 1
            assert label != ""

        shutil.rmtree(self.tempdir)

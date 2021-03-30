import os
import json
import sys
import pathlib
import tempfile
import io
from contextlib import redirect_stdout

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase


class TestPrintResults(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.out_dir1 = pathlib.Path(tempfile.mkdtemp(prefix="hi"))
        self.out_dir2 = pathlib.Path(tempfile.mkdtemp(prefix="hi"))

        self.directory1 = self.TEST_DIR / "results1"
        self.directory2 = self.TEST_DIR / "results2"
        self.directory3 = self.TEST_DIR / "results3"
        os.makedirs(self.directory1)
        os.makedirs(self.directory2)
        os.makedirs(self.directory3)
        json.dump(
            {"train": 1, "test": 2, "dev": 3},
            open(os.path.join(self.directory1 / "metrics.json"), "w+"),
        )
        json.dump(
            {"train": 4, "dev": 5}, open(os.path.join(self.directory2 / "metrics.json"), "w+")
        )
        json.dump(
            {"train": 6, "dev": 7}, open(os.path.join(self.directory3 / "cool_metrics.json"), "w+")
        )

    def test_print_results(self):
        kebab_args = [
            "__main__.py",
            "print-results",
            str(self.TEST_DIR),
            "--keys",
            "train",
            "dev",
            "test",
        ]
        sys.argv = kebab_args
        with io.StringIO() as buf, redirect_stdout(buf):
            main()
            output = buf.getvalue()

        lines = output.strip().split("\n")
        assert lines[0] == "model_run, train, dev, test"

        expected_results = {
            (str(self.directory1) + "/metrics.json", "1", "3", "2"),
            (str(self.directory2) + "/metrics.json", "4", "5", "N/A"),
        }
        results = {tuple(line.split(", ")) for line in lines[1:]}
        assert results == expected_results

    def test_print_results_with_metrics_filename(self):
        kebab_args = [
            "__main__.py",
            "print-results",
            str(self.TEST_DIR),
            "--keys",
            "train",
            "dev",
            "test",
            "--metrics-filename",
            "cool_metrics.json",
        ]
        sys.argv = kebab_args
        with io.StringIO() as buf, redirect_stdout(buf):
            main()
            output = buf.getvalue()

        lines = output.strip().split("\n")
        assert lines[0] == "model_run, train, dev, test"

        expected_results = {(str(self.directory3) + "/cool_metrics.json", "6", "7", "N/A")}
        results = {tuple(line.split(", ")) for line in lines[1:]}
        assert results == expected_results

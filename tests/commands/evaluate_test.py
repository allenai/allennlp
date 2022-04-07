import argparse
import json
from pathlib import Path
from typing import Iterator, List, Dict
from shutil import copyfile
import pytest
import torch
from flaky import flaky

from allennlp.commands.evaluate import evaluate_from_args, Evaluate
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.data_loaders import TensorDict
from allennlp.models import Model


class DummyDataLoader:
    def __init__(self, outputs: List[TensorDict]) -> None:
        super().__init__()
        self._outputs = outputs

    def __iter__(self) -> Iterator[TensorDict]:
        yield from self._outputs

    def __len__(self):
        return len(self._outputs)

    def set_target_device(self, _):
        pass


class DummyModel(Model):
    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore
        return kwargs


class TestEvaluate(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title="Commands", metavar="")
        Evaluate().add_subparser(subparsers)

    @flaky
    def test_evaluate_from_args(self):
        kebab_args = [
            "evaluate",
            str(
                self.FIXTURES_ROOT / "simple_tagger_with_span_f1" / "serialization" / "model.tar.gz"
            ),
            str(self.FIXTURES_ROOT / "data" / "conll2003.txt"),
            "--cuda-device",
            "-1",
        ]

        args = self.parser.parse_args(kebab_args)
        metrics = evaluate_from_args(args)
        assert metrics.keys() == {
            "accuracy",
            "accuracy3",
            "precision-overall",
            "recall-overall",
            "f1-measure-overall",
            "loss",
        }

    def test_output_file_evaluate_from_args(self):
        output_file = str(self.TEST_DIR / "metrics.json")
        predictions_output_file = str(self.TEST_DIR / "predictions.jsonl")
        kebab_args = [
            "evaluate",
            str(
                self.FIXTURES_ROOT / "simple_tagger_with_span_f1" / "serialization" / "model.tar.gz"
            ),
            str(self.FIXTURES_ROOT / "data" / "conll2003.txt"),
            "--cuda-device",
            "-1",
            "--output-file",
            output_file,
            "--predictions-output-file",
            predictions_output_file,
        ]
        args = self.parser.parse_args(kebab_args)
        computed_metrics = evaluate_from_args(args)

        with open(output_file, "r") as file:
            saved_metrics = json.load(file)
        assert computed_metrics == saved_metrics

        with open(predictions_output_file, "r") as file:
            for line in file:
                prediction = json.loads(line.strip())
            assert "tags" in prediction

    def test_multiple_output_files_evaluate_from_args(self):
        data_file = Path(self.FIXTURES_ROOT / "data" / "conll2003.txt")
        paths = []
        out_paths = []
        pred_paths = []
        for i in range(3):
            tmp_path = self.TEST_DIR.joinpath(f"TEST{i}.txt")

            # Need to create paths to check when they do not exist
            out_paths.append(tmp_path.parent.joinpath(f"OUTPUTS{i}.json"))
            pred_paths.append(tmp_path.parent.joinpath(f"PREDS{i}.txt"))

            copyfile(data_file, tmp_path)
            paths.append(tmp_path)

        kebab_args = [
            "evaluate",
            str(
                self.FIXTURES_ROOT / "simple_tagger_with_span_f1" / "serialization" / "model.tar.gz"
            ),
            ",".join(map(str, paths)),
            "--cuda-device",
            "-1",
            "--output-file",
            ",".join(map(str, out_paths)),
            "--predictions-output-file",
            ",".join(map(str, pred_paths)),
        ]
        args = self.parser.parse_args(kebab_args)
        computed_metrics = evaluate_from_args(args)
        computed_by_file = {}
        for k, v in computed_metrics.items():
            fn, *metric_name = k.split("_")
            if fn not in computed_by_file:
                computed_by_file[fn] = {}
            computed_by_file[fn]["_".join(metric_name)] = v

        assert len(computed_by_file) == len(paths)
        expected_input_data = data_file.read_text("utf-8")

        for i, p in enumerate(paths):
            # Make sure it was not modified
            assert p.read_text("utf-8") == expected_input_data

            assert p.stem in computed_by_file, f"paths[{i}]={p.stem}"

            assert out_paths[i].exists(), f"paths[{i}]={p.stem}"
            saved_metrics = json.loads(out_paths[i].read_text("utf-8"))
            assert saved_metrics == computed_by_file[p.stem], f"paths[{i}]={p.stem}"
            assert pred_paths[i].exists(), f"paths[{i}]={p.stem}"

    def test_evaluate_works_with_vocab_expansion(self):
        archive_path = str(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        # snli2 has a extra token ("seahorse") in it.
        evaluate_data_path = str(
            self.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_corpus2.jsonl"
        )
        embeddings_filename = str(
            self.FIXTURES_ROOT / "data" / "unawarded_embeddings.gz"
        )  # has only unawarded vector
        embedding_sources_mapping = json.dumps(
            {"_text_field_embedder.token_embedder_tokens": embeddings_filename}
        )
        kebab_args = ["evaluate", archive_path, evaluate_data_path, "--cuda-device", "-1"]

        # TODO(mattg): the unawarded_embeddings.gz file above doesn't exist, but this test still
        # passes. This suggests that vocab extension in evaluate isn't currently doing anything,
        # and so it is broken.

        # Evaluate 1 with no vocab expansion,
        # Evaluate 2 with vocab expansion with no pretrained embedding file.
        # Evaluate 3 with vocab expansion with given pretrained embedding file.
        metrics_1 = evaluate_from_args(self.parser.parse_args(kebab_args))
        metrics_2 = evaluate_from_args(self.parser.parse_args(kebab_args + ["--extend-vocab"]))
        metrics_3 = evaluate_from_args(
            self.parser.parse_args(
                kebab_args + ["--embedding-sources-mapping", embedding_sources_mapping]
            )
        )
        assert metrics_1 != metrics_2
        assert metrics_2 != metrics_3

    @pytest.mark.parametrize("auto_names", ["NONE", "METRICS", "PREDS", "ALL"])
    def test_auto_names_creates_files(self, auto_names):
        data_file = Path(self.FIXTURES_ROOT / "data" / "conll2003.txt")
        paths = []
        out_paths = []
        pred_paths = []
        for i in range(5):
            tmp_path = self.TEST_DIR.joinpath(f"TEST{i}.txt")

            # Need to create paths to check when they do not exist
            out_paths.append(tmp_path.parent.joinpath(f"OUTPUTS{i}.json"))
            pred_paths.append(tmp_path.parent.joinpath(f"PREDS{i}.txt"))

            copyfile(data_file, tmp_path)
            paths.append(tmp_path)

        kebab_args = [
            "evaluate",
            str(
                self.FIXTURES_ROOT / "simple_tagger_with_span_f1" / "serialization" / "model.tar.gz"
            ),
            ",".join(map(str, paths)),
            "--cuda-device",
            "-1",
            "--output-file",
            ",".join(map(str, out_paths)),
            "--predictions-output-file",
            ",".join(map(str, pred_paths)),
            "--auto-names",
            auto_names,
        ]

        args = self.parser.parse_args(kebab_args)
        _ = evaluate_from_args(args)

        expected_input_data = data_file.read_text("utf-8")

        for i, p in enumerate(paths):
            # Make sure it was not modified
            assert p.read_text("utf-8") == expected_input_data

            if auto_names == "METRICS" or auto_names == "ALL":
                assert not out_paths[i].exists()
                assert p.parent.joinpath(f"{p.stem}.outputs").exists()
            else:
                assert out_paths[i].exists()
                assert not p.parent.joinpath(f"{p.stem}.outputs").exists()
            if auto_names == "PREDS" or auto_names == "ALL":
                assert not pred_paths[i].exists()
                assert p.parent.joinpath(f"{p.stem}.preds").exists()
            else:
                assert pred_paths[i].exists()
                assert not p.parent.joinpath(f"{p.stem}.preds").exists()

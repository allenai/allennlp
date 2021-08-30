import os
import pathlib
import tempfile
from allennlp.commands.train import train_model_from_file

import allennlp_models.mc.models
import allennlp_models.mc.dataset_readers
import torch

TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")
TEST_DIR = pathlib.Path(TEST_DIR)
os.makedirs(TEST_DIR, exist_ok=True)


def bench_no_torchscript(benchmark):
    param_file = os.path.join("torchscript", "experiment.jsonnet")
    save_dir = TEST_DIR / "torchscript" / "save_dir"
    # train_model_from_file(param_file, save_dir)
    benchmark(train_model_from_file, param_file, save_dir, force=True)


def bench_with_torchscript(benchmark):
    param_file = os.path.join("torchscript", "experiment.jsonnet")
    save_dir = TEST_DIR / "torchscript" / "save_dir"
    # train_model_from_file(param_file, save_dir)
    benchmark(train_model_from_file, param_file, save_dir, force=True)

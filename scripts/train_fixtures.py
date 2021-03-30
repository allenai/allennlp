#!/usr/bin/env python

import glob
import logging
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.commands.test_install import _get_module_root
from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params
from allennlp.common.util import pushd


logger = logging.getLogger(__name__)


def train_fixture(config_prefix: str, config_filename: str = "experiment.json") -> None:
    config_file = config_prefix + config_filename
    serialization_dir = config_prefix + "serialization"
    # Train model doesn't like it if we have incomplete serialization
    # directories, so remove them if they exist.
    if os.path.exists(serialization_dir):
        shutil.rmtree(serialization_dir)

    # train the model
    train_model_from_file(config_file, serialization_dir)

    # remove unnecessary files
    shutil.rmtree(os.path.join(serialization_dir, "log"))

    for filename in glob.glob(os.path.join(serialization_dir, "*")):
        if (
            filename.endswith(".log")
            or filename.endswith(".json")
            or re.search(r"epoch_[0-9]+\.th$", filename)
        ):
            os.remove(filename)


def train_fixture_gpu(config_prefix: str) -> None:
    config_file = config_prefix + "experiment.json"
    serialization_dir = config_prefix + "serialization"
    params = Params.from_file(config_file)
    params["trainer"]["cuda_device"] = 0

    # train this one to a tempdir
    tempdir = tempfile.gettempdir()
    train_model(params, tempdir)

    # now copy back the weights and and archived model
    shutil.copy(os.path.join(tempdir, "best.th"), os.path.join(serialization_dir, "best_gpu.th"))
    shutil.copy(
        os.path.join(tempdir, "model.tar.gz"), os.path.join(serialization_dir, "model_gpu.tar.gz")
    )


if __name__ == "__main__":
    module_root = _get_module_root().parent
    with pushd(module_root, verbose=True):
        models = [
            ("basic_classifier", "experiment_seq2seq.jsonnet"),
            "simple_tagger",
            "simple_tagger_with_elmo",
            "simple_tagger_with_span_f1",
        ]
        for model in models:
            if isinstance(model, tuple):
                model, config_filename = model
                train_fixture(f"allennlp/tests/fixtures/{model}/", config_filename)
            else:
                train_fixture(f"allennlp/tests/fixtures/{model}/")

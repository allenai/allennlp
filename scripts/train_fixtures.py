#!/usr/bin/env python

import re
import os
import glob
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params

def train_fixture(config_file: str, serialization_dir: str) -> None:
    # train the model
    train_model_from_file(config_file, serialization_dir)

    # remove unnecessary files
    shutil.rmtree(os.path.join(serialization_dir, "log"))

    for filename in glob.glob(os.path.join(serialization_dir, "*")):
        if filename.endswith(".log") or filename.endswith(".json") or re.search(r"epoch_[0-9]+\.th$", filename):
            os.remove(filename)

def train_fixture_gpu(config_file: str, serialization_dir: str) -> None:
    params = Params.from_file(config_file)
    params["trainer"]["cuda_device"] = 0

    # train this one to a tempdir
    tempdir = tempfile.gettempdir()
    train_model(params, tempdir)

    # now copy back the weights and and archived model
    shutil.copy(os.path.join(tempdir, "best.th"), os.path.join(serialization_dir, "best_gpu.th"))
    shutil.copy(os.path.join(tempdir, "model.tar.gz"), os.path.join(serialization_dir, "model_gpu.tar.gz"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "gpu":
        train_fixture_gpu("tests/fixtures/srl/experiment.json", "tests/fixtures/srl/serialization")
    else:
        train_fixture("tests/fixtures/decomposable_attention/experiment.json", "tests/fixtures/decomposable_attention/serialization")
        train_fixture("tests/fixtures/bidaf/experiment.json", "tests/fixtures/bidaf/serialization")
        train_fixture("tests/fixtures/srl/experiment.json", "tests/fixtures/srl/serialization")
        train_fixture("tests/fixtures/coref/experiment.json", "tests/fixtures/coref/serialization")

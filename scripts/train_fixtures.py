#!/usr/bin/env python

import re
import os
import glob
import shutil
import sys
import tempfile

from allennlp.commands.train import train_model_from_file, train_model, _CONFIG_FILE_KEY
from allennlp.common import Params

def train_fixture(config_file: str) -> None:
    params = Params.from_file(config_file)
    serialization_dir = params.get("trainer").get("serialization_dir")

    # train the model
    train_model_from_file(config_file)

    # remove unnecessary files
    shutil.rmtree(os.path.join(serialization_dir, "log"))

    for filename in glob.glob(os.path.join(serialization_dir, "*")):
        if filename.endswith(".log") or filename.endswith(".json") or re.search(r"epoch_[0-9]+\.th$", filename):
            os.remove(filename)

def train_fixture_gpu(config_file: str) -> None:
    params = Params.from_file(config_file)
    params[_CONFIG_FILE_KEY] = config_file
    params["trainer"]["cuda_device"] = 0

    # train this one to a tempdir
    serialization_dir = params["trainer"]["serialization_dir"]
    tempdir = tempfile.gettempdir()
    params["trainer"]["serialization_dir"] = tempdir

    train_model(params)

    # now copy back the weights and and archived model
    shutil.copy(os.path.join(tempdir, "best.th"), os.path.join(serialization_dir, "best_gpu.th"))
    shutil.copy(os.path.join(tempdir, "model.tar.gz"), os.path.join(serialization_dir, "model_gpu.tar.gz"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "gpu":
        train_fixture_gpu("tests/fixtures/srl/experiment.json")
    else:
        train_fixture("tests/fixtures/decomposable_attention/experiment.json")
        train_fixture("tests/fixtures/bidaf/experiment.json")
        train_fixture("tests/fixtures/srl/experiment.json")

#!/usr/bin/env python
import logging
import os
import sys

import torch

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)

from allennlp.commands import main  # noqa


def run():
    main(prog="allennlp")


if __name__ == "__main__":
    # First, let Pytorch's multiprocessing module know how to create child processes.
    # Refer https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing.set_start_method
    torch.multiprocessing.set_start_method("spawn")

    run()

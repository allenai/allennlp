#!/usr/bin/env python
import argparse
import logging

from allennlp.common import Params
from allennlp.training.trainer import TrainerPieces

#Needed?
#sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--serialization-dir", required=True)
    args = parser.parse_args()

    params = Params.from_file(args.config)
    pieces = TrainerPieces.from_params(params, args.serialization_dir)

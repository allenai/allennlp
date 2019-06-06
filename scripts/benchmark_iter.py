#!/usr/bin/env python
import argparse
import logging

from allennlp.common import Params, Tqdm
from allennlp.training.trainer import TrainerPieces

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--serialization-dir", required=True)
    args = parser.parse_args()

    params = Params.from_file(args.config)
    pieces = TrainerPieces.from_params(params, args.serialization_dir)

    # Get tqdm for the training batches
    raw_generator = pieces.iterator(pieces.train_dataset,
                                      num_epochs=1,
                                      shuffle=True)
    generator_tqdm = Tqdm.tqdm(raw_generator)
    for batch in generator_tqdm:
        pass

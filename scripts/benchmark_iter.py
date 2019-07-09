#!/usr/bin/env python
import argparse
import logging
import time

from allennlp.common import Params, Tqdm
from allennlp.training.trainer_pieces import TrainerPieces

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

BATCH_INTERVAL = 100

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

    start = time.perf_counter()
    last = start
    batch_count = 0
    cumulative_batch_size = 0
    for batch in generator_tqdm:
        batch_count += 1
        cumulative_batch_size += batch['source']['tokens'].size(0)

        if batch_count % BATCH_INTERVAL == 0:
            end = time.perf_counter()
            print(f"b/s total: {(end - start)/batch_count} b/s last: {(end - last)/BATCH_INTERVAL} mean batch size: {cumulative_batch_size/batch_count}")
            last = end

        pass

#!/usr/bin/env python
import argparse
import logging
import time

from allennlp.common import Params, Tqdm
from allennlp.training.trainer_pieces import TrainerPieces

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

BATCH_INTERVAL = 100
# On subset of 1b word corpus
MEAN_BATCH_SIZE = 66.0


def time_iterable(iterable, get_size, batch_interval):
    start = time.perf_counter()
    last = start
    batch_count = 0
    cumulative_batch_size = 0
    for batch in iterable:
        batch_count += 1
        cumulative_batch_size += get_size(batch)

        if batch_count % batch_interval == 0:
            end = time.perf_counter()
            approximate_intervals = batch_count / batch_interval
            print(f"b/s total: {(end - start)/batch_count} b/s last: {(end - last)/batch_interval} mean interval size: {cumulative_batch_size/approximate_intervals}")
            last = end

        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--serialization-dir", required=True)
    args = parser.parse_args()

    params = Params.from_file(args.config)
    pieces = TrainerPieces.from_params(params, args.serialization_dir)

    # Time just the reader.
    #time_iterable(pieces.train_dataset, lambda batch: 1, BATCH_INTERVAL * MEAN_BATCH_SIZE)

    # Get tqdm for the training batches
    raw_generator = pieces.iterator(pieces.train_dataset,
                                      num_epochs=1,
                                      shuffle=True)
    generator_tqdm = Tqdm.tqdm(raw_generator)
    time_iterable(generator_tqdm, lambda batch: batch['source']['tokens'].size(0), BATCH_INTERVAL)


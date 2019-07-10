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


def time_iterable(iterable, get_items_per_batch):
    start = time.perf_counter()
    last = start

    item_count = 0
    for batch in iterable:
        item_count += get_items_per_batch(batch)
        adjusted_batch_count = item_count / MEAN_BATCH_SIZE

        if 0 <= adjusted_batch_count % BATCH_INTERVAL < 1:
            end = time.perf_counter()

            msg = (f"s/b total: {(end - start) / adjusted_batch_count} " +
                   f"s/b last: {(end - last) / BATCH_INTERVAL} "
                   f"adjusted batch count: {adjusted_batch_count}")
            print(msg)

            last = end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--serialization-dir", required=True)
    args = parser.parse_args()

    params = Params.from_file(args.config)
    pieces = TrainerPieces.from_params(params, args.serialization_dir)

    # Time just the reader.
    #time_iterable(pieces.train_dataset, lambda batch: 1)

    # Get tqdm for the training batches
    raw_generator = pieces.iterator(pieces.train_dataset,
                                      num_epochs=1,
                                      shuffle=True)
    generator_tqdm = Tqdm.tqdm(raw_generator)
    time_iterable(generator_tqdm, lambda batch: batch['source']['tokens'].size(0))


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


def time_iterable(iterable, get_items_per_batch, batches_per_interval):
    start = time.perf_counter()
    last = start

    batch_count = 0
    item_count = 0
    for batch in iterable:
        batch_count += 1
        item_count += get_items_per_batch(batch)
        adjusted_batch_count = item_count / MEAN_BATCH_SIZE

        if batch_count % batches_per_interval == 0:
            end = time.perf_counter()
            # With this sleep I actually see items in the output queue.
            #time.sleep(10)
            # Conclusion: Tensorizing is slow?

            #import inspect
            #inspect.getmembers
            #iterable.gi_frame.f_locals['self'].output_queue
            #import pdb;pdb.set_trace()
            msg = (f"s/b total: {(end - start) / adjusted_batch_count:.3f} " +
                   f"s/b last: {(end - last) / BATCH_INTERVAL:.3f} " +
                   f"read out q: {iterable.gi_frame.f_locals['instances'].output_queue.qsize()} " +
                   f"it in q: {iterable.gi_frame.f_locals['input_queue'].qsize()} " +
                   f"it out q: {iterable.gi_frame.f_locals['output_queue'].qsize()} " +
                   f"~ batches: {adjusted_batch_count:.1f}")
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
    #time_iterable(pieces.train_dataset, lambda batch: 1, BATCH_INTERVAL * MEAN_BATCH_SIZE)

    # Get tqdm for the training batches
    raw_generator = pieces.iterator(pieces.train_dataset,
                                      num_epochs=1,
                                      shuffle=True)
    time_iterable(raw_generator, lambda batch: batch['source']['tokens'].size(0), BATCH_INTERVAL)


#!/usr/bin/env python

# Benchmarks the iterator (and indirectly the dataset reader) for a given config.
#
# Example 1: Log stats every 100 batches. Periodically output internals of
# MultiprocessDatasetReader and MultiprocessIterator.
#
# $ scripts/benchmark_iter.py --config training_config/bidirectional_language_model.jsonnet --serialization-dir serialization-dir --action=log --assume-multiprocess-types
#
# Example 2: Output seconds/batch over 10k batches.
#
# $ scripts/benchmark_iter.py --config training_config/bidirectional_language_model.jsonnet --serialization-dir serialization-dir --action=time --batch-count=10000
#
# Example 3: Output seconds to produce the first batch in order to measure overhead.
#
# $ scripts/benchmark_iter.py --config training_config/bidirectional_language_model.jsonnet --serialization-dir serialization-dir --action=first


import argparse
from enum import Enum
import logging
from multiprocessing import Process
import time

from allennlp.common import Params, Tqdm
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.util import get_batch_size

BATCH_INTERVAL = 100
LOGGING_INTERVAL_SECONDS = 5

def run_periodically(reader_output, iterator_output):
    while True:
        message = (f"read out q: {reader_output.qsize()} " +
                   f"it out q: {iterator_output.qsize()}")
        print(message)
        time.sleep(LOGGING_INTERVAL_SECONDS)

def log_iterable(iterable, assume_multiprocess_types):
    start = time.perf_counter()
    last = start
    periodic_logging_process = None
    have_started_periodic_process = False

    batch_count = 0
    cumulative_batch_size = 0
    cumulative_token_count = 0
    for batch in iterable:
        batch_count += 1
        cumulative_batch_size += get_batch_size(batch)
        tokens_size = batch['source']['tokens'].size()
        cumulative_token_count += tokens_size[0] * tokens_size[1]

        if assume_multiprocess_types and not have_started_periodic_process:
            have_started_periodic_process = True
            periodic_logging_process = Process(
                    target=run_periodically,
                    # Pass the queues directly. Passing the iterable naively
                    # won't work because the forked process (in contrast with
                    # threads) has an entirely separate address space.
                    # Presumably this could be worked around with
                    # multiprocessing.managers or similar.
                    args=(iterable.gi_frame.f_locals['qiterable'].output_queue,
                          iterable.gi_frame.f_locals['output_queue']
                    )
                    )
            periodic_logging_process.start()

        if batch_count % BATCH_INTERVAL == 0:
            end = time.perf_counter()

            msg = (f"s/b total: {(end - start) / batch_count:.3f} " +
                   f"s/b last: {(end - last) / BATCH_INTERVAL:.3f} " +
                   f"batch count: {batch_count} " +
                   f"batch size: {cumulative_batch_size / batch_count:.1f} " +
                   f"total tokens {cumulative_token_count}")
            print(msg)

            last = end

    if periodic_logging_process:
        periodic_logging_process.terminate()

def time_iterable(iterable, batch_count):
    assert batch_count > 0

    print("Starting test")
    start = time.perf_counter()

    i = batch_count
    for _ in iterable:
        i -= 1
        if i == 0:
            break
    assert i == 0, "Not enough batches!"

    end = time.perf_counter()
    print(f"{(end - start)/batch_count:.3f} s/b over {batch_count} batches")

def time_to_first(iterable):
    print("Starting test")
    start = time.perf_counter()

    for _ in iterable:
        break

    end = time.perf_counter()
    print(f"{(end - start):.3f} s/b for first batch")

class Action(Enum):
    log = "log"
    time = "time"
    first = "first"

    def __str__(self):
        return self.name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=Action, choices=list(Action), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--serialization-dir", required=True)
    parser.add_argument("--batch-count", type=int, default=0)
    parser.add_argument("--assume-multiprocess-types", action="store_true")
    args = parser.parse_args()

    params = Params.from_file(args.config)
    pieces = TrainerPieces.from_params(params, args.serialization_dir)

    raw_generator = pieces.iterator(pieces.train_dataset,
                                    num_epochs=1,
                                    shuffle=True)

    if args.action is Action.log:
        log_iterable(raw_generator, args.assume_multiprocess_types)
    elif args.action is Action.time:
        time_iterable(raw_generator, args.batch_count)
    elif args.action is Action.first:
        time_to_first(raw_generator)
    else:
        raise Exception(f"Unaccounted for action {action}")


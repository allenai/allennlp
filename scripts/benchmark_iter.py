#!/usr/bin/env python
import argparse
from enum import Enum
import logging
from multiprocessing import Process
import time

from allennlp.common import Params, Tqdm
from allennlp.training.trainer_pieces import TrainerPieces

BATCH_INTERVAL = 100
# On subset of 1b word corpus
MEAN_BATCH_SIZE = 66.0
LOGGING_INTERVAL_SECONDS = 10

def run_periodically(reader_output,
        #iterator_input,
        iterator_output):
    while True:
        message = (f"read out q: {reader_output.qsize()} " +
                  #f"it in q: {iterator_input.qsize()} " +
                  f"it out q: {iterator_output.qsize()}")
        print(message)
        time.sleep(10)

def log_iterable(iterable, get_items_per_batch, batches_per_interval):
    start = time.perf_counter()
    last = start
    periodic_logging_process = None
    have_started_periodic_process = False

    batch_count = 0
    item_count = 0
    for batch in iterable:
        if not have_started_periodic_process:
            have_started_periodic_process = True
            periodic_logging_process = Process(
                    target=run_periodically,
                    # Pass the queues directly. Passing the iterable naively
                    # won't work because the forked process (in contrast with
                    # threads) has an entirely separate address space.
                    # Presumably this could be worked around with
                    # multiprocessing.managers or similar.
                    #args=(iterable.gi_frame.f_locals['instances'].output_queue,
                    #      iterable.gi_frame.f_locals['input_queue'],
                    #      iterable.gi_frame.f_locals['output_queue']
                    #)
                    args=(iterable.gi_frame.f_locals['qiterable'].output_queue,
                          iterable.gi_frame.f_locals['output_queue']
                    )
                    )
            periodic_logging_process.start()
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
            # TODO(brendanr): Put the queue output on a timer.
            # TODO(brendanr): Have a mode where we don't log anything for timing purposes. We just chug through 10k batches.
            msg = (f"s/b total: {(end - start) / adjusted_batch_count:.3f} " +
                   f"s/b last: {(end - last) / BATCH_INTERVAL:.3f} " +
                   f"~ batches: {adjusted_batch_count:.1f}")
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
    args = parser.parse_args()

    params = Params.from_file(args.config)
    pieces = TrainerPieces.from_params(params, args.serialization_dir)

    raw_generator = pieces.iterator(pieces.train_dataset,
                                    num_epochs=1,
                                    shuffle=True)

    if args.action is Action.log:
        log_iterable(raw_generator, lambda batch: batch['source']['tokens'].size(0), BATCH_INTERVAL)
    elif args.action is Action.time:
        time_iterable(raw_generator, args.batch_count)
    elif args.action is Action.first:
        time_to_first(raw_generator)
    else:
        raise Exception(f"Unaccounted for action {action}")


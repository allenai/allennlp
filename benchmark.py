
from typing import Tuple
import time
import argparse

import numpy
from torch.utils.data import Dataset, DataLoader

from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator, DataIterator, BasicIterator, MultiprocessIterator
from allennlp.data.dataset_readers import LanguageModelingReader, DatasetReader, MultiprocessDatasetReader, SnliReader


"""
Case 1: 
    Dataset is in-memory, pre-indexed and pre-tensorised.
    This is not the standard way of iteration in allennlp.

Case 2:
    Dataset is in-memory, pre-indexed, but not pre-tensorised.

Case 3:
    Dataset is in-memory, not indexed, and not tensorised.
    This is the standard way of iteration in allennlp.

Case 4:
    Dataset is lazily read from disk, not indexed, and not tensorised.
    This is the standard way of iteration in allennlp if
    the lazy=True flag is passed.


In each of these cases, we would like to benchmark the following iterators:

1. same process, BasicIterator

2. multiple processes for batching data, MultiProcessIterator

3. multiple processes for batching data,
   multiple processes for reading data from disk, 
   MultiProcessIterator + MultiProcessDatasetReader

4. Native pytorch Dataset, DataLoader.

"""

class SlowSnli(SnliReader):


    def text_to_instance(self, *args, **kwargs):

        time.sleep(0.01)
        return super().text_to_instance(*args, **kwargs)


class AllennlpDataset(Dataset):

    def __init__(self,
                 vocab: Vocabulary,
                 reader: DatasetReader,
                 dataset_path: str):

        self.vocab = vocab
        self.reader = reader
        self.dataset_path = dataset_path

        self.iterable = self.reader.read(dataset_path)
        self.iterator = (x for x in self.iterable)


        # MASSIVE HACK! 
        # In real life, we will use a IterableDataset,
        # so we won't have this length problem. This is also
        # fine in terms of measurement, because we are still reading
        # the dataset from disk - just this time, we know apriori how
        # long it is.
        self._length = None
        if "1000" in dataset_path:
            self._length = 1000
        if "5000" in dataset_path:
            self._length = 5000
        if "10000" in dataset_path:
            self._length = 10000


    def __len__(self):
        """
        This is gross but will go away in the next pytorch release,
        as they are introducing an `IterableDataset` class which doesn't
        need to have a length:
        https://pytorch.org/docs/master/data.html#torch.utils.data.IterableDataset
        """
        if self._length is None:
            self._length = 0
            for i in self.iterator:
                self._length += 1
            self.iterator = (x for x in self.iterable)
        return self._length

    def __getitem__(self, idx) -> Instance:
        get_next = next(self.iterator, None)
        if get_next is None:
            self.iterator = (x for x in self.iterable)
            get_next = next(self.iterator)
        return get_next

    def allennlp_collocate(self, batch):
        batch = Batch(batch)
        batch.index_instances(self.vocab)
        return batch.as_tensor_dict(batch.get_padding_lengths())



BATCH_SIZE = 32
DATA_PATH = "/Users/markn/allen_ai/data/snli_1.0/1000.jsonl"

def get_base_reader_and_vocab(in_memory: bool,
                              num_workers: int = None) -> Tuple[DatasetReader, Vocabulary]:

    reader = SnliReader(lazy=not in_memory)
    # Create vocab before making the reader multiprocess, for convienience.
    instance_iterable = reader.read(DATA_PATH)
    vocab = Vocabulary.from_instances(instance_iterable)
    #reader = SlowSnli(lazy=not in_memory)

    if num_workers is not None:
        return MultiprocessDatasetReader(reader, num_workers=num_workers), vocab

    return reader, vocab


def get_iterator(num_workers: int = None) -> DataIterator:

    iterator = BasicIterator(batch_size=BATCH_SIZE)
    if num_workers is not None:
        return MultiprocessIterator(iterator, num_workers)

    return iterator


def measure_performance(in_memory: bool,
                        pre_indexed: bool, 
                        pre_tensorised: bool,
                        reader_workers: int = None,
                        iterator_workers: int = None,
                        pytorch_loader: bool = False
                        ):

    # Make super, duper sure we aren't running a configuration which doesn't make sense.
    if not in_memory:
        assert pre_indexed is False, "A lazy dataset cannot be pre-indexed."
        assert pre_tensorised is False, "A lazy dataset cannot be pre-tensorised."

    if pre_tensorised:
        assert pre_indexed, "a dataset cannot be pre-tensorised but not pre-indexed"

    if reader_workers is not None:
        assert not in_memory, "A multiprocess reader is inherrently lazy."
        assert not pre_indexed, "We cannot preindex a dataset which is not in memory."
        assert not pre_tensorised, "We cannot pretensorise a dataset which is not in memory."

    reader, vocab = get_base_reader_and_vocab(in_memory, reader_workers)

    if not pytorch_loader:

        iterator = get_iterator(iterator_workers)
        # This could be 3 things:
        # 1. A list of instances.
        # 2. An iterable of instances.
        # 3. A QIterable of instances.
        instances = reader.read(DATA_PATH)

        # Give the multiprocess dataset reader time to start chewing
        if reader_workers is not None:
            time.sleep(8)

        if in_memory:
            if pre_indexed:
                # This indexes the instances in place.
                Batch(instances).index_instances(vocab)
            else:
                iterator.index_with(vocab)

            if pre_tensorised:
                # in the pre-tensorised case, we actually want the lists of 
                # tensor dicts. So the most comparable way to do that is to
                # just run through the iterator once, and collect the tensors.
                instances = [x for x in iterator(instances, num_epochs=1, shuffle=False)]

        else:
            iterator.index_with(vocab)

        if pre_tensorised:
            batch_generator = instances
        else:
            batch_generator = iterator(instances, num_epochs=1, shuffle=False)

        if iterator_workers is not None:
            # Give the multiprocess iterator time to start up and fill it's queues etc.
            time.sleep(8)

    else:
        dataset = AllennlpDataset(vocab, reader, DATA_PATH)
        batch_generator = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=iterator_workers or 0, collate_fn=dataset.allennlp_collocate)
        time.sleep(8)

    t1 = time.time()
    i = 0
    for batch in batch_generator:
        i += 1

        # Allennlp multiprocessing code returns more batches, not sure why
        if i == 30:
            break
    print(f"iterated over {i} batches")
    t2 = time.time()

    overall = t2 - t1
    return overall

def non_multiprocessing_sanity(repeats):
    # Non multiprocessing.
    fastest_possible =  [measure_performance(in_memory=True, pre_indexed=True, pre_tensorised=True) for t in range(repeats)]
    not_tensorised = [measure_performance(in_memory=True, pre_indexed=True, pre_tensorised=False) for t in range(repeats)]
    not_indexed = [measure_performance(in_memory=True, pre_indexed=False, pre_tensorised=False) for t in range(repeats)]
    lazy = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False) for t in range(repeats)]
    pytorch_lazy = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, pytorch_loader=True) for t in range(repeats)]

    print(f"in memory, pre indexed, pre tensorised: \t\t", numpy.mean(fastest_possible), numpy.std(fastest_possible))
    print(f"in memory, pre indexed, not tensorised: \t\t", numpy.mean(not_tensorised), numpy.std(not_tensorised))
    print(f"in memory, not indexed, not tensorised: \t\t", numpy.mean(not_indexed), numpy.std(not_indexed))
    print(f"lazy, not indexed, not tensorised: \t\t\t", numpy.mean(lazy), numpy.std(lazy))
    print(f"pytorch loader lazy, not indexed, not tensorised: \t\t\t", numpy.mean(pytorch_lazy), numpy.std(pytorch_lazy))


def full_multiprocessing_comparison(repeats):

    baseline = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False) for t in range(repeats)]
    baseline_pytorch = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, pytorch_loader=True) for t in range(repeats)]
    two_workers = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, iterator_workers=2, reader_workers=2) for t in range(repeats)]
    five_workers = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, iterator_workers=5, reader_workers=5) for t in range(repeats)]
    #ten_workers = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, iterator_workers=10, reader_workers=10) for t in range(repeats)]
    pytorch_loader_two_workers = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, iterator_workers=2, pytorch_loader=True) for t in range(repeats)] 
    pytorch_loader_five_workers = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, iterator_workers=5, pytorch_loader=True) for t in range(repeats)] 
    #pytorch_loader_ten_workers = [measure_performance(in_memory=False, pre_indexed=False, pre_tensorised=False, iterator_workers=10, pytorch_loader=True) for t in range(repeats)] 
    print(f"Full Multiprocessing Worker Comparison")
    print(f"baseline: \t\t\t",  numpy.mean(baseline), numpy.std(baseline))
    print(f"pytorch, baseline: \t\t\t",  numpy.mean(baseline_pytorch), numpy.std(baseline_pytorch))
    print(f"two workers: \t\t\t",  numpy.mean(two_workers), numpy.std(two_workers))
    print(f"five workers: \t\t\t",  numpy.mean(five_workers), numpy.std(five_workers))
    #print(f"ten workers: \t\t\t",  numpy.mean(ten_workers), numpy.std(ten_workers))
    print(f"pytorch, two workers: \t\t\t",  numpy.mean(pytorch_loader_two_workers), numpy.std(pytorch_loader_two_workers))
    print(f"pytorch, five workers: \t\t\t",  numpy.mean(pytorch_loader_five_workers), numpy.std(pytorch_loader_five_workers))
    #print(f"pytorch, ten workers: \t\t\t",  numpy.mean(pytorch_loader_ten_workers), numpy.std(pytorch_loader_ten_workers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()

    if args.sanity:
        non_multiprocessing_sanity(5)

    if args.multi:
        full_multiprocessing_comparison(3)
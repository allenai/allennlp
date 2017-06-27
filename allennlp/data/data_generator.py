from typing import List, Tuple
import logging
import random
from copy import deepcopy

from ..common.params import Params
from ..common.util import group_by_count, add_noise_to_dict_values
from . import Dataset, Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataGenerator:
    """
    A ``DataGenerator`` takes an :class:`~.dataset.IndexedDataset` and converts it into a
    generator, yielding batches suitable for training.  You might want to do this instead of just
    creating one large set of numpy arrays for a few reasons:

    #. Creating large arrays for your whole data could take a whole lot of memory, maybe more than
       is available on your machine.
    #. Creating one large array means padding all of your instances to the same length.  This
       typically means you waste a whole lot of computation on padding tokens.  Using a
       ``DataGenerator`` instead allows you to only pad each `batch` to the same length, instead of
       all of your instances across your whole dataset.  We've typically seen a 4-5x speed up just
       from doing this (partially because Keras is pretty bad at doing variable-length computation;
       the speed-up isn't quite as large with plain tensorflow, I think).
    #. If we're varying the padding lengths in each batch, we can also vary the batch size, to
       optimize GPU memory usage.  This means we'll use smaller batch sizes for big instances, and
       larger batch sizes for small instances.  We've seen speedups up to 10-12x (on top of the
       4-5x speed up above) from doing this.

    Parameters
    ----------
    text_trainer: TextTrainer
        We need access to the ``TextTrainer`` object so we can call some methods on it, such as
        :func:`~allennlp.training.TextTrainer.get_instance_sorting_keys`.
    dynamic_padding: bool, optional (default=False)
        If ``True``, we will set padding lengths based on the data `per batch`, instead of on the
        whole dataset.  This only works if your model is structured to allow variable-length
        sequences (typically using ``None`` for specific dimensions when you build your model), and
        if you don't set padding values in
        :func:`~allennlp.training.TextTrainer._set_padding_lengths`.  This flag specifically is read
        in :func:`~allennlp.training.TextTrainer._set_padding_lengths` to know if we should set
        certain padding values or not.  It's handled correctly for ``num_sentence_words`` and
        ``num_word_characters`` in :class:`~allennlp.training.TextTrainer`, but you need to be sure
        to implement it correctly in subclasses for this to work.
    padding_noise: double, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    sort_every_epoch: bool, optional (default=True)
        If ``True``, we will re-sort the data after every epoch, then re-group the instances into
        batches.  If ``padding_noise`` is zero, this does nothing, but if it's non-zero, this will
        give you a slightly different ordering, so you don't have exactly the same batches at every
        epoch.  If you're doing adaptive batch sizes, this will lead to re-computing the adaptive
        batches each epoch, which could give a different number of batches for the whole dataset,
        which means each "epoch" might no longer correspond to `exactly` one pass over the data.
        This is probably a pretty minor issue, though.
    adaptive_batch_sizes: bool, optional (default=False)
        Only relevant if ``dynamic_padding`` is ``True``.  If ``adaptive_batch_sizes`` is ``True``,
        we will vary the batch size to try to optimize GPU memory usage.  Because padding lengths
        are done dynamically, we can have larger batches when padding lengths are smaller,
        maximizing our usage of the GPU.  In order for this to work, you need to do two things: (1)
        override :func:`~TextTrainer._get_padding_memory_scaling` to give a big-O bound on memory
        usage given padding lengths, and (2) tune the `adaptive_memory_usage_constant` parameter
        for your particular model and GPU.  See the documentation for
        :func:`~TextTrainer._get_padding_memory_scaling` for more information.
    adaptive_memory_usage_constant: int, optional (default=None)
        Only relevant if ``adaptive_batch_sizes`` is ``True``.  This is a manually-tuned parameter,
        specific to a particular model architecture and amount of GPU memory (e.g., if you change
        the number of hidden layers in your model, this number will need to change).  See
        :func:`~TextTrainer._get_padding_memory_scaling` for more detail.  The recommended way to
        tune this parameter is to (1) use a fixed batch size, with ``biggest_batch_first`` set to
        ``True``, and find out the maximum batch size you can handle on your biggest instances
        without running out of memory.  Then (2) turn on ``adaptive_batch_sizes``, and set this
        parameter so that you get the right batch size for your biggest instances.  If you set the
        log level to ``DEBUG`` in ``scripts/run_model.py``, you can see the batch sizes that are
        computed.
    maximum_batch_size: int, optional (default=1000000)
        If we're using adaptive batch sizes, you can use this to be sure you do not create batches
        larger than this, even if you have enough memory to handle it on your GPU.  You might
        choose to do this to keep smaller batches because you like the noisier gradient estimates
        that come from smaller batches, for instance.
    biggest_batch_first: bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        It's only meaningful if you're using dynamic padding - this will let you try out the
        largest batch that you have in the data `first`, so that if you're going to run out of
        memory, you know it early, instead of waiting through the whole batch to find out at the
        end that you're going to crash.
    """
    def __init__(self, text_trainer, params: Params):
        self.text_trainer = text_trainer
        self.dynamic_padding = params.pop('dynamic_padding', False)
        self.padding_noise = params.pop('padding_noise', 0.2)
        self.sort_every_epoch = params.pop('sort_every_epoch', True)
        self.adaptive_batch_sizes = params.pop('adaptive_batch_sizes', False)
        self.adaptive_memory_usage_constant = params.pop('adaptive_memory_usage_constant', False)
        self.maximum_batch_size = params.pop('maximum_batch_size', 1000000)
        self.biggest_batch_first = params.pop('biggest_batch_first', False)

        #: This field can be read after calling ``create_generator`` to get the number of steps you
        #: should take per epoch in ``model.fit_generator`` or ``model.evaluate_generator`` for
        #: this data.
        self.last_num_batches = None

    def create_generator(self, dataset: Dataset, batch_size: int=None):
        """
        Main external API call: converts an ``IndexedDataset`` into a data generator suitable for
        use with Keras' ``fit_generator`` and related methods.
        """
        if batch_size is None:
            batch_size = self.text_trainer.batch_size

        grouped_instances = self.__create_batches(dataset, batch_size)
        self.last_num_batches = len(grouped_instances)
        def generator():
            while True:
                if self.sort_every_epoch:
                    unpadded_dataset = deepcopy(dataset)
                    groups = self.__create_batches(unpadded_dataset, batch_size)
                else:
                    groups = grouped_instances
                for group in groups:
                    batch = Dataset(group)

                    yield batch.as_arrays(self.text_trainer.get_padding_lengths(), verbose=False)
        return generator()

    def __create_batches(self, dataset: Dataset, batch_size: int) -> List[List[Instance]]:
        instances = dataset.instances
        if self.dynamic_padding:
            instances = self.sort_dataset_by_padding(dataset,
                                                     self.text_trainer.get_instance_sorting_keys(),
                                                     self.padding_noise)
        if self.adaptive_batch_sizes:
            grouped_instances = self.__adaptive_grouping(instances)
        else:
            grouped_instances = group_by_count(instances, batch_size, None)
            grouped_instances[-1] = [instance for instance in grouped_instances[-1] if instance is not None]
        if self.biggest_batch_first:
            # We'll actually pop the last _two_ batches, because the last one might not
            # be full.
            last_batch = grouped_instances.pop()
            penultimate_batch = grouped_instances.pop()
            random.shuffle(grouped_instances)
            grouped_instances.insert(0, penultimate_batch)
            grouped_instances.insert(0, last_batch)
        else:
            random.shuffle(grouped_instances)
        return grouped_instances

    def __adaptive_grouping(self, instances: List[Instance]):
        batches = []
        current_batch = []
        current_lengths = {}
        logger.debug("Creating adatpive groups")
        for instance in instances:
            current_batch.append(instance)
            instance_lengths = instance.get_padding_lengths()
            for key in instance_lengths:
                current_lengths[key] = max(instance_lengths[key], current_lengths.get(key, -1))
            big_o_memory_constant = self.text_trainer.get_padding_memory_scaling(current_lengths)
            if (len(current_batch) * big_o_memory_constant > self.adaptive_memory_usage_constant
                        or len(current_batch) > self.maximum_batch_size):
                current_batch.pop()
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    padding_lengths = Dataset(current_batch).get_padding_lengths()
                    logger.debug("Batch size: %d; padding: %s", len(current_batch), padding_lengths)
                batches.append(current_batch)
                current_batch = [instance]
                current_lengths = instance_lengths
        if logger.getEffectiveLevel() <= logging.DEBUG:
            padding_lengths = Dataset(current_batch).get_padding_lengths()
            logger.debug("Batch size: %d; padding: %s", len(current_batch), padding_lengths)
        batches.append(current_batch)
        return batches

    @staticmethod
    def sort_dataset_by_padding(dataset: Dataset,
                                sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                                padding_noise: float=0.0) -> List[Instance]:
        """
        Sorts the ``Instances`` in this ``Dataset`` by their padding lengths, using the keys in
        ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
        ``(field_name, padding_key)`` tuples.
        """
        instances_with_lengths = []
        for instance in dataset.instances:
            padding_lengths = instance.get_padding_lengths()
            if padding_noise > 0.0:
                noisy_lengths = {}
                for field_name, field_lengths in padding_lengths:
                    noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
                padding_lengths = noisy_lengths
            instance_with_lengths = [padding_lengths[field_name][padding_key]
                                     for (field_name, padding_key) in sorting_keys] + [instance]
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[:-1])
        return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

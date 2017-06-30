from typing import Callable, Dict, List, Tuple
import random
import logging

from overrides import overrides
from ..instance import Instance
from ..dataset import Dataset
from .bucket_iterator import BucketIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AdaptiveIterator(BucketIterator):
    """
    Parameters
    ----------
    adaptive_memory_usage_constant : int, required.
        Only relevant if ``use_adaptive_grouping`` is ``True``.  This is a manually-tuned parameter,
        specific to a particular model architecture and amount of GPU memory (e.g., if you change
        the number of hidden layers in your model, this number will need to change). The recommended
        way to tune this parameter is to (1) use a fixed batch size, with ``biggest_batch_first``
        set to ``True``, and find out the maximum batch size you can handle on your biggest instances
        without running out of memory.  Then (2) turn on ``use_adaptive_grouping``, and set this
        parameter so that you get the right batch size for your biggest instances.  If you set the
        log level to ``DEBUG`` in ``scripts/run_model.py``, you can see the batch sizes that are
        computed.
    padding_memory_scaling: Callable[Dict[str, int], float], required.
        This function is used for computing the adaptive batch sizes.  We assume that memory usage is a
        function that looks like this: :math:`M = b * O(p) * c`, where :math:`M` is the memory
        usage, :math:`b` is the batch size, :math:`c` is some constant that depends on how much GPU
        memory you have and various model hyperparameters, and :math:`O(p)` is a function outlining
        how memory usage asymptotically varies with the padding lengths.  Our approach will be to
        let the user effectively set :math:`\\frac{M}{c}` using the ``adaptive_memory_usage_constant``
        above. This function specifies :math:`O(p)`, so we can solve for the batch size :math:`b`.
        The more specific you get in specifying :math:`O(p)` in this function, the better a job we
        can do in optimizing memory usage.
    maximum_batch_size : int, optional (default=10000)
        If we're using adaptive batch sizes, you can use this to be sure you do not create batches
        larger than this, even if you have enough memory to handle it on your GPU.  You might
        choose to do this to keep smaller batches because you like the noisier gradient estimates
        that come from smaller batches, for instance.
    use_adaptive_grouping : bool, optional (default = True)
        If ``use_adaptive_grouping`` is ``True``, we will vary the batch size to try to optimize
        GPU memory usage.  Because padding lengths are done dynamically, we can have larger batches
        when padding lengths are smaller, maximizing our usage of the GPU.  In order for this to work,
         you need to do two things:
        (1) Provide the ``padding_memory_scaling`` function, which gives a big-O bound on memory
        usage given padding lengths. For instance, if you have two TextFields with ``sentence_lengths``
        which require padding, this might be simply |sentence1| * |sentence2|.
        (2) Tune the `adaptive_memory_usage_constant` parameter for your particular model and GPU.
        The only reason this is a parameter is so that you can do this tuning easily by setting this to
        False. If you are not using adaptive batch sizes, you should use a :class:`~BucketIterator`.
    biggest_batch_first: bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        It's only meaningful if you're using dynamic padding - this will let you try out the
        largest batch that you have in the data `first`, so that if you're going to run out of
        memory, you know it early, instead of waiting through the whole batch to find out at the
        end that you're going to crash.

    See :class:`~BucketIterator` for a description of the other parameters.
    """
    def __init__(self,
                 adaptive_memory_usage_constant: float,
                 padding_memory_scaling: Callable[Dict[str, int], float],
                 maximum_batch_size: int = 10000,
                 use_adaptive_grouping: bool = True,
                 biggest_batch_first: bool = False,
                 sorting_keys: List[Tuple[str, str]] = None,
                 padding_noise: float = 0.2,
                 sort_every_epoch: bool = True):

        self.padding_memory_scaling = padding_memory_scaling
        self.maximum_batch_size = maximum_batch_size
        self.adaptive_memory_usage_constant = adaptive_memory_usage_constant
        self.biggest_batch_first = biggest_batch_first
        self.use_adaptive_grouping = use_adaptive_grouping
        super(AdaptiveIterator, self).__init__(sorting_keys, padding_noise, sort_every_epoch)

    @overrides
    def _create_batches(self, dataset: Dataset) -> List[List[Instance]]:

        if self.use_adaptive_grouping:
            if self.sorting_keys:
                instances = self.sort_dataset_by_padding(dataset,
                                                         self.sorting_keys,
                                                         self.padding_noise)
            else:
                instances = dataset.instances
            # Group the instances into different sized batches,
            # depending on how padded they are.
            grouped_instances = self.__adaptive_grouping(instances)
        else:
            grouped_instances = super(AdaptiveIterator, self)._create_batches(dataset)
        if self.biggest_batch_first:
            # We'll actually pop the last _two_ batches,
            # because the last one might not be full.
            last_batch = grouped_instances.pop()
            penultimate_batch = grouped_instances.pop()
            random.shuffle(grouped_instances)
            grouped_instances.insert(0, penultimate_batch)
            grouped_instances.insert(0, last_batch)
        else:
            random.shuffle(grouped_instances)
        return grouped_instances

    def __adaptive_grouping(self, dataset: Dataset):
        batches = []
        current_batch = []
        current_lengths = {}
        logger.debug("Creating adaptive groups")
        for instance in dataset.instances:
            current_batch.append(instance)
            instance_lengths = instance.get_padding_lengths()
            for key in instance_lengths:
                current_lengths[key] = max(instance_lengths[key], current_lengths.get(key, -1))
            big_o_memory_constant = self.padding_memory_scaling(current_lengths)
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

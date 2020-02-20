"""
The various :class:`~allennlp.data.iterators.data_iterator.DataIterator` subclasses
can be used to iterate over datasets with different batching and padding schemes.
"""

from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.homogeneous_batch_iterator import HomogeneousBatchIterator
from allennlp.data.iterators.multiprocess_iterator import MultiprocessIterator
from allennlp.data.iterators.pass_through_iterator import PassThroughIterator

from collections import OrderedDict

from .data_iterator import DataIterator
from .basic_iterator import BasicIterator
from .bucket_iterator import BucketIterator
from .adaptive_iterator import AdaptiveIterator


iterators = OrderedDict()  # pylint: disable=invalid-name
iterators["bucket"] = BucketIterator
iterators["basic"] = BasicIterator
iterators["adaptive"] = AdaptiveIterator

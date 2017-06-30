from collections import OrderedDict

from .basic_iterator import BasicIterator
from .bucket_iterator import BucketIterator
from .adaptive_iterator import AdaptiveIterator


iterators = OrderedDict()
iterators["bucket"] = BucketIterator
iterators["basic"] = BasicIterator
iterators[""]
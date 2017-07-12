from collections import OrderedDict
from typing import Dict, cast, Type

from .data_iterator import DataIterator
from .basic_iterator import BasicIterator
from .bucket_iterator import BucketIterator
from .adaptive_iterator import AdaptiveIterator

# pylint: disable=invalid-name
iterators = OrderedDict()  # type: Dict[str, Type[DataIterator]]
# pylint: enable=invalid-name

iterators["bucket"] = BucketIterator
iterators["basic"] = BasicIterator
iterators["adaptive"] = AdaptiveIterator

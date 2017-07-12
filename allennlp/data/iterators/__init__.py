from collections import OrderedDict
from typing import Dict, cast

from .data_iterator import DataIterator
from .basic_iterator import BasicIterator
from .bucket_iterator import BucketIterator
from .adaptive_iterator import AdaptiveIterator

# pylint: disable=invalid-name
iterators = OrderedDict()  # type: Dict[str, 'DataIterator']
# pylint: enable=invalid-name

iterators["bucket"] = cast(DataIterator, BucketIterator)
iterators["basic"] = cast(DataIterator, BasicIterator)
iterators["adaptive"] = cast(DataIterator, AdaptiveIterator)

from collections import OrderedDict
from typing import Dict, Type

from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.adaptive_iterator import AdaptiveIterator

# pylint: disable=invalid-name
iterators = OrderedDict()  # type: Dict[str, Type[DataIterator]]
# pylint: enable=invalid-name

iterators["bucket"] = BucketIterator
iterators["basic"] = BasicIterator
iterators["adaptive"] = AdaptiveIterator

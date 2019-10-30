from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, MultiprocessDatasetReader
from allennlp.data.iterators import MultiprocessIterator, TransformIterator, DataIterator
from allennlp.data.iterators.basic_iterator import BasicIteratorStub
from allennlp.data import transforms
from allennlp.data.vocabulary import Vocabulary
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class TestMultiprocessIterator(IteratorTest):
    def test_construction_returns_modified_base_iterator(self):
        iterator = MultiprocessIterator(BasicIteratorStub(batch_size=32), num_workers=3)
        assert isinstance(iterator, TransformIterator)
        assert isinstance(iterator.transforms[-1], transforms.Fork)
        assert iterator._num_workers == 3

    def test_construction_returns_modified_base_iterator_from_params(self):

        params = Params(
            {
                "type": "multiprocess",
                "base_iterator": {"type": "basic", "batch_size": 32},
                "num_workers": 3,
            }
        )
        iterator = DataIterator.from_params(params)
        assert isinstance(iterator, TransformIterator)
        assert isinstance(iterator.transforms[-1], transforms.Fork)
        assert iterator._num_workers == 3

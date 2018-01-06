# pylint: disable=no-self-use,invalid-name
from typing import List

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import LazyDataset, Instance, Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.iterators import LazyIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer

class LazyIteratorTestCase(AllenNlpTestCase):
    def setUp(self):
        super(LazyIteratorTestCase, self).setUp()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.vocab = Vocabulary()
        self.this_index = self.vocab.add_token_to_namespace('this')
        self.is_index = self.vocab.add_token_to_namespace('is')
        self.a_index = self.vocab.add_token_to_namespace('a')
        self.sentence_index = self.vocab.add_token_to_namespace('sentence')
        self.another_index = self.vocab.add_token_to_namespace('another')
        self.yet_index = self.vocab.add_token_to_namespace('yet')
        self.very_index = self.vocab.add_token_to_namespace('very')
        self.long_index = self.vocab.add_token_to_namespace('long')
        self.instances = [
                self.create_instance(["this", "is", "a", "sentence"]),
                self.create_instance(["this", "is", "another", "sentence"]),
                self.create_instance(["yet", "another", "sentence"]),
                self.create_instance(["this", "is", "a", "very", "very", "very", "very", "long", "sentence"]),
                self.create_instance(["sentence"]),
                ]
        self.dataset = LazyDataset(lambda: (i for i in self.instances))
        self.dataset.index_instances(self.vocab)

    def create_instance(self, str_tokens: List[str]):
        tokens = [Token(t) for t in str_tokens]
        instance = Instance({'text': TextField(tokens, self.token_indexers)})
        instance.index_fields(self.vocab)
        return instance

    def assert_instances_are_correct(self, candidate_instances):
        # First we need to remove padding tokens from the candidates.
        # pylint: disable=protected-access
        candidate_instances = [tuple(w for w in instance if w != 0) for instance in candidate_instances]
        expected_instances = [tuple(instance.fields["text"]._indexed_tokens["tokens"])
                              for instance in self.instances]
        assert set(candidate_instances) == set(expected_instances)


class TestLazyIterator(LazyIteratorTestCase):
    # We also test some of the stuff in `DataIterator` here.
    def test_yield_one_epoch_iterates_over_the_data_once(self):
        iterator = LazyIterator(batch_size=2)
        batches = list(iterator(self.dataset, num_epochs=1, shuffle=False))
        # We just want to get the single-token array for the text field in the instance.
        instances = [tuple(instance.data.cpu().numpy())
                     for batch in batches
                     for instance in batch['text']["tokens"]]
        assert len(instances) == 5
        self.assert_instances_are_correct(instances)

    def test_call_iterates_over_data_forever(self):
        generator = LazyIterator(batch_size=2)(self.dataset, shuffle=False)
        batches = [next(generator) for _ in range(18)]  # going over the data 6 times
        # We just want to get the single-token array for the text field in the instance.
        instances = [tuple(instance.data.cpu().numpy())
                     for batch in batches
                     for instance in batch['text']["tokens"]]
        assert len(instances) == 5 * 6
        self.assert_instances_are_correct(instances)

    def test_from_params(self):
        # pylint: disable=protected-access
        params = Params({})
        iterator = LazyIterator.from_params(params)
        assert iterator._batch_size == 32  # default value

        params = Params({"batch_size": 10})
        iterator = LazyIterator.from_params(params)
        assert iterator._batch_size == 10

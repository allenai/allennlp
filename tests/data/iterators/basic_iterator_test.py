# pylint: disable=no-self-use,invalid-name
from typing import List

from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.testing.test_case import AllenNlpTestCase


class TestBasicIterator(AllenNlpTestCase):
    def setUp(self):
        super(TestBasicIterator, self).setUp()
        self.token_indexers = [SingleIdTokenIndexer()]
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
        self.dataset = Dataset(self.instances)

    def create_instance(self, tokens: List[str]):
        instance = Instance({'text': TextField(tokens, self.token_indexers)})
        instance.index_fields(self.vocab)
        return instance

    def assert_instances_are_correct(self, candidate_instances):
        # First we need to remove padding tokens from the candidates.
        # pylint: disable=protected-access
        candidate_instances = [tuple(w for w in instance if w != 0) for instance in candidate_instances]
        expected_instances = [tuple(instance.get_field("text")._indexed_tokens[0]) for instance in self.instances]
        print(candidate_instances)
        print(expected_instances)
        assert set(candidate_instances) == set(expected_instances)

    def test_num_batches_per_epoch_calculates_correctly(self):
        iterator = BasicIterator(batch_size=2)
        assert iterator.num_batches_per_epoch(self.dataset) == 3

    def test_yield_one_pass_iterates_over_the_data_once(self):
        iterator = BasicIterator(batch_size=2)
        batches = list(iterator.yield_one_pass(self.dataset))
        # We just want to get the single-token array for the text field in the instance.
        instances = [tuple(instance) for batch in batches for instance in batch['text'][0]]
        assert len(instances) == 5
        self.assert_instances_are_correct(instances)

    def test_call_iterates_over_data_forever(self):
        generator = BasicIterator(batch_size=2)(self.dataset)
        batches = [next(generator) for _ in range(18)]  # going over the data 6 times
        # We just want to get the single-token array for the text field in the instance.
        instances = [tuple(instance) for batch in batches for instance in batch['text'][0]]
        assert len(instances) == 5 * 6
        self.assert_instances_are_correct(instances)

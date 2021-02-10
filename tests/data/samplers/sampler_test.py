from typing import List, Iterable, Dict, Union

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary, Instance, Token, Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class LazyIterable:
    def __init__(self, instances):
        self._instances = instances

    def __iter__(self):
        return (instance for instance in self._instances)


class SamplerTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.vocab = Vocabulary()
        self.this_index = self.vocab.add_token_to_namespace("this")
        self.is_index = self.vocab.add_token_to_namespace("is")
        self.a_index = self.vocab.add_token_to_namespace("a")
        self.sentence_index = self.vocab.add_token_to_namespace("sentence")
        self.another_index = self.vocab.add_token_to_namespace("another")
        self.yet_index = self.vocab.add_token_to_namespace("yet")
        self.very_index = self.vocab.add_token_to_namespace("very")
        self.long_index = self.vocab.add_token_to_namespace("long")
        instances = [
            self.create_instance(["this", "is", "a", "sentence"]),
            self.create_instance(["this", "is", "another", "sentence"]),
            self.create_instance(["yet", "another", "sentence"]),
            self.create_instance(
                ["this", "is", "a", "very", "very", "very", "very", "long", "sentence"]
            ),
            self.create_instance(["sentence"]),
        ]

        self.instances = instances
        self.lazy_instances = LazyIterable(instances)

    def create_instance(self, str_tokens: List[str]):
        tokens = [Token(t) for t in str_tokens]
        instance = Instance({"text": TextField(tokens, self.token_indexers)})
        return instance

    def create_instances_from_token_counts(self, token_counts: List[int]) -> List[Instance]:
        return [self.create_instance(["word"] * count) for count in token_counts]

    def get_batches_stats(self, batches: Iterable[Batch]) -> Dict[str, Union[int, List[int]]]:
        grouped_instances = [batch.instances for batch in batches]
        group_lengths = [len(group) for group in grouped_instances]

        sample_sizes = []
        for batch in batches:
            batch_sequence_length = max(
                instance.get_padding_lengths()["text"]["tokens___tokens"]
                for instance in batch.instances
            )
            sample_sizes.append(batch_sequence_length * len(batch.instances))

        return {
            "batch_lengths": group_lengths,
            "total_instances": sum(group_lengths),
            "sample_sizes": sample_sizes,
        }

    def assert_instances_are_correct(self, candidate_instances):
        # First we need to remove padding tokens from the candidates.

        candidate_instances = [
            tuple(w for w in instance if w != 0) for instance in candidate_instances
        ]
        expected_instances = [
            tuple(instance.fields["text"]._indexed_tokens["tokens"]["tokens"])
            for instance in self.instances
        ]
        assert set(candidate_instances) == set(expected_instances)

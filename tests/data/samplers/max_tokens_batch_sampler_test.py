from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.samplers import MaxTokensBatchSampler
from allennlp.data.data_loaders import MultiProcessDataLoader

from .sampler_test import SamplerTest


class TestMaxTokensSampler(SamplerTest):
    def test_create_batches_groups_correctly(self):
        sampler = MaxTokensBatchSampler(max_tokens=8, padding_noise=0, sorting_keys=["text"])

        grouped_instances = []
        for indices in sampler.get_batch_indices(self.instances):
            grouped_instances.append([self.instances[idx] for idx in indices])
        expected_groups = [
            [self.instances[4], self.instances[2]],
            [self.instances[0], self.instances[1]],
            [self.instances[3]],
        ]
        for group in grouped_instances:
            assert group in expected_groups
            expected_groups.remove(group)
        assert expected_groups == []

    def test_guess_sorting_key_picks_the_longest_key(self):
        sampler = MaxTokensBatchSampler(max_tokens=8, padding_noise=0)
        instances = []
        short_tokens = [Token(t) for t in ["what", "is", "this", "?"]]
        long_tokens = [Token(t) for t in ["this", "is", "a", "not", "very", "long", "passage"]]
        instances.append(
            Instance(
                {
                    "question": TextField(short_tokens, self.token_indexers),
                    "passage": TextField(long_tokens, self.token_indexers),
                }
            )
        )
        instances.append(
            Instance(
                {
                    "question": TextField(short_tokens, self.token_indexers),
                    "passage": TextField(long_tokens, self.token_indexers),
                }
            )
        )
        instances.append(
            Instance(
                {
                    "question": TextField(short_tokens, self.token_indexers),
                    "passage": TextField(long_tokens, self.token_indexers),
                }
            )
        )
        assert sampler.sorting_keys is None
        sampler._guess_sorting_keys(instances)
        assert sampler.sorting_keys == ["passage"]

    def test_batch_count(self):
        sampler = MaxTokensBatchSampler(max_tokens=8, padding_noise=0, sorting_keys=["text"])
        data_loader = MultiProcessDataLoader(
            self.get_mock_reader(), "fake_path", batch_sampler=sampler
        )
        assert len(data_loader) == 3

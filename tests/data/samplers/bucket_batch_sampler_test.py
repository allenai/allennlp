from allennlp.common import Params
from allennlp.data import Instance, Token, Batch
from allennlp.data.fields import TextField
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.data_loaders import MultiProcessDataLoader

from .sampler_test import SamplerTest


class TestBucketSampler(SamplerTest):
    def test_create_batches_groups_correctly(self):
        sampler = BucketBatchSampler(batch_size=2, padding_noise=0, sorting_keys=["text"])

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

    def test_disable_shuffle(self):
        sampler = BucketBatchSampler(batch_size=2, sorting_keys=["text"], shuffle=False)

        grouped_instances = []
        for indices in sampler.get_batch_indices(self.instances):
            grouped_instances.append([self.instances[idx] for idx in indices])
        expected_groups = [
            [self.instances[4], self.instances[2]],
            [self.instances[0], self.instances[1]],
            [self.instances[3]],
        ]
        for idx, group in enumerate(grouped_instances):
            assert group == expected_groups[idx]

    def test_guess_sorting_key_picks_the_longest_key(self):
        sampler = BucketBatchSampler(batch_size=2, padding_noise=0)
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

    def test_from_params(self):
        params = Params({})

        sorting_keys = ["s1", "s2"]
        params["sorting_keys"] = sorting_keys
        params["batch_size"] = 32
        sampler = BucketBatchSampler.from_params(params=params)

        assert sampler.sorting_keys == sorting_keys
        assert sampler.padding_noise == 0.1
        assert sampler.batch_size == 32

        params = Params(
            {
                "sorting_keys": sorting_keys,
                "padding_noise": 0.5,
                "batch_size": 100,
                "drop_last": True,
            }
        )

        sampler = BucketBatchSampler.from_params(params=params)
        assert sampler.sorting_keys == sorting_keys
        assert sampler.padding_noise == 0.5
        assert sampler.batch_size == 100
        assert sampler.drop_last

    def test_drop_last_works(self):
        sampler = BucketBatchSampler(
            batch_size=2,
            padding_noise=0,
            sorting_keys=["text"],
            drop_last=True,
        )

        # We use a custom collate_fn for testing, which doesn't actually create tensors,
        # just the allennlp Batches.
        def collate_fn(x, **kwargs):
            return Batch(x)

        data_loader = MultiProcessDataLoader(
            self.get_mock_reader(),
            "fake_path",
            batch_sampler=sampler,
        )
        data_loader.collate_fn = collate_fn
        data_loader.index_with(self.vocab)
        batches = [batch for batch in iter(data_loader)]
        stats = self.get_batches_stats(batches)

        # all batches have length batch_size
        assert all(batch_len == 2 for batch_len in stats["batch_lengths"])

        # we should have lost one instance by skipping the last batch
        assert stats["total_instances"] == len(self.instances) - 1

    def test_batch_count(self):
        sampler = BucketBatchSampler(batch_size=2, padding_noise=0, sorting_keys=["text"])
        data_loader = MultiProcessDataLoader(
            self.get_mock_reader(), "fake_path", batch_sampler=sampler
        )
        data_loader.index_with(self.vocab)
        assert len(data_loader) == 3

    def test_batch_count_with_drop_last(self):
        sampler = BucketBatchSampler(
            batch_size=2,
            padding_noise=0,
            sorting_keys=["text"],
            drop_last=True,
        )
        data_loader = MultiProcessDataLoader(
            self.get_mock_reader(), "fake_path", batch_sampler=sampler
        )
        assert len(data_loader) == 2

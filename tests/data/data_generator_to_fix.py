# pylint: disable=no-self-use,invalid-name
import numpy

from allennlp.common.params import Params
from allennlp.data.data_generator import DataGenerator
from allennlp.testing.test_case import DeepQaTestCase


class TestDataGenerator(DeepQaTestCase):
    def setUp(self):
        super(TestDataGenerator, self).setUp()
        self.text_trainer = FakeTextTrainer()
        self.instances = [
                FakeInstance(0, 5, 3, 2),
                FakeInstance(1, 4, 3, 2),
                FakeInstance(2, 4, 1, 2),
                FakeInstance(3, 9, 3, 2),
                FakeInstance(4, 8, 3, 2),
                FakeInstance(5, 2, 1, 2),
                FakeInstance(6, 3, 3, 2),
                FakeInstance(7, 3, 3, 3),
                FakeInstance(8, 1, 1, 2),
                FakeInstance(9, 1, 1, 3),
                ]

    def test_instances_are_sorted_by_sorting_keys(self):
        params = Params({
                'dynamic_padding': True,
                'padding_noise': 0.0,
                })
        generator = DataGenerator(self.text_trainer, params)
        batches = generator.create_generator(IndexedDataset(self.instances))
        assert generator.last_num_batches == 4
        one_epoch_arrays = [next(batches) for _ in range(4)]
        one_epoch_arrays.sort(key=lambda x: x[0][0])
        assert self.as_list(one_epoch_arrays[0][0]) == [1, 0, 4]
        assert self.as_list(one_epoch_arrays[1][0]) == [3]
        assert self.as_list(one_epoch_arrays[2][0]) == [6, 7, 2]
        assert self.as_list(one_epoch_arrays[3][0]) == [8, 9, 5]

    def test_batches_are_consistent_with_no_repermuting(self):
        params = Params({
                'padding_noise': 0.0,
                'sort_every_epoch': False,
                'dynamic_padding': True,
                })
        generator = DataGenerator(self.text_trainer, params)
        batches = generator.create_generator(IndexedDataset(self.instances))
        assert generator.last_num_batches == 4
        first_epoch_arrays = [next(batches) for _ in range(4)]
        second_epoch_arrays = [next(batches) for _ in range(4)]
        first_epoch_arrays.sort(key=lambda x: x[0][0])
        second_epoch_arrays.sort(key=lambda x: x[0][0])
        first_epoch = [self.as_list(x[0]) for x in first_epoch_arrays]
        second_epoch = [self.as_list(x[0]) for x in second_epoch_arrays]
        assert first_epoch == second_epoch

    def test_biggest_batch_first(self):
        params = Params({
                'padding_noise': 0.0,
                'dynamic_padding': True,
                'biggest_batch_first': True,
                })
        generator = DataGenerator(self.text_trainer, params)
        batches = generator.create_generator(IndexedDataset(self.instances))
        biggest_batches = [next(batches) for _ in range(2)]
        assert self.as_list(biggest_batches[0][0]) == [3]
        assert self.as_list(biggest_batches[1][0]) == [1, 0, 4]

    def test_adaptive_grouping(self):
        params = Params({
                'padding_noise': 0.0,
                'dynamic_padding': True,
                'adaptive_batch_sizes': True,
                'adaptive_memory_usage_constant': 130,
                })
        generator = DataGenerator(self.text_trainer, params)
        batches = generator.create_generator(IndexedDataset(self.instances))
        assert generator.last_num_batches == 4
        one_epoch_arrays = [next(batches) for _ in range(4)]
        one_epoch_arrays.sort(key=lambda x: x[0][0])
        assert self.as_list(one_epoch_arrays[0][0]) == [0, 4]
        assert self.as_list(one_epoch_arrays[1][0]) == [3]
        assert self.as_list(one_epoch_arrays[2][0]) == [7, 2, 1]
        assert self.as_list(one_epoch_arrays[3][0]) == [8, 9, 5, 6]

    def test_sort_every_batch_actually_adds_noise_every_batch(self):
        # We're just going to get two epoch's worth of batches, and make sure that they're
        # different.
        params = Params({
                'padding_noise': 0.8,
                'sort_every_epoch': True,
                'dynamic_padding': True,
                })
        generator = DataGenerator(self.text_trainer, params)
        batches = generator.create_generator(IndexedDataset(self.instances))
        assert generator.last_num_batches == 4
        first_epoch_arrays = [next(batches) for _ in range(4)]
        second_epoch_arrays = [next(batches) for _ in range(4)]
        first_epoch_arrays.sort(key=lambda x: x[0][0])
        second_epoch_arrays.sort(key=lambda x: x[0][0])
        first_epoch = [self.as_list(x[0]) for x in first_epoch_arrays]
        second_epoch = [self.as_list(x[0]) for x in second_epoch_arrays]
        assert first_epoch != second_epoch

    def test_maximum_batch_size_is_actually_a_maximum(self):
        params = Params({
                'padding_noise': 0.0,
                'dynamic_padding': True,
                'adaptive_batch_sizes': True,
                'adaptive_memory_usage_constant': 50,
                'maximum_batch_size': 2,
                })
        generator = DataGenerator(self.text_trainer, params)
        batches = generator.create_generator(IndexedDataset(self.instances))
        assert generator.last_num_batches == 7
        one_epoch_arrays = [next(batches) for _ in range(7)]
        one_epoch_arrays.sort(key=lambda x: x[0][0])
        print([self.as_list(x[0]) for x in one_epoch_arrays])
        assert self.as_list(one_epoch_arrays[0][0]) == [0]
        assert self.as_list(one_epoch_arrays[1][0]) == [2, 1]
        assert self.as_list(one_epoch_arrays[2][0]) == [3]
        assert self.as_list(one_epoch_arrays[3][0]) == [4]
        assert self.as_list(one_epoch_arrays[4][0]) == [5, 6]
        assert self.as_list(one_epoch_arrays[5][0]) == [7]
        assert self.as_list(one_epoch_arrays[6][0]) == [8, 9]

    def as_list(self, array):
        return list(numpy.squeeze(array, axis=-1))


class FakeInstance:
    def __init__(self, index, a_length, b_length, c_length):
        self.index = index
        self.a_length = a_length
        self.b_length = b_length
        self.c_length = c_length

    def get_padding_lengths(self):
        return {'a': self.a_length, 'b': self.b_length, 'c': self.c_length}

    def pad(self, lengths):
        pass

    def as_training_data(self):
        return numpy.asarray([self.index]), numpy.asarray([self.index])


class FakeTextTrainer:
    batch_size = 3
    a_length = None
    b_length = None
    c_length = None
    def get_instance_sorting_keys(self):
        return ['a', 'b', 'c']

    def get_padding_lengths(self):
        return {'a': self.a_length, 'b': self.b_length, 'c': self.c_length}

    def get_padding_memory_scaling(self, lengths):
        return lengths['a'] * lengths['b'] * lengths['c']

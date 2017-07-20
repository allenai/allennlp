# pylint: disable=invalid-name, no-self-use
import numpy
import torch
import pytest

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.common.tensor import arrays_to_variables


class TestTensor(AllenNlpTestCase):

    def test_data_structure_as_variables_handles_recursion(self):

        array_dict = {
                "sentence": {
                        "words": numpy.zeros([3, 4]),
                        "characters": numpy.ones([2, 5])
                        },
                "tags": numpy.ones([2, 3])
        }
        torch_array_dict = arrays_to_variables(array_dict)

        assert torch_array_dict["sentence"]["words"].data.equal(
                torch.DoubleTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].data.equal(
                torch.DoubleTensor(numpy.ones([2, 5])))
        assert torch_array_dict["tags"].data.equal(
                torch.DoubleTensor(numpy.ones([2, 3])))

    def test_data_structure_as_variables_correctly_converts_mixed_types(self):

        array_dict = {
                "sentence": {
                        "words": numpy.zeros([3, 4], dtype="float32"),
                        "characters": numpy.ones([2, 5], dtype="int32")
                        },
                "tags": numpy.ones([2, 3], dtype="uint8")
        }
        torch_array_dict = arrays_to_variables(array_dict)

        assert torch_array_dict["sentence"]["words"].data.equal(
                torch.FloatTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].data.equal(
                torch.IntTensor(numpy.ones([2, 5], dtype="int32")))
        assert torch_array_dict["tags"].data.equal(torch.ByteTensor(
                numpy.ones([2, 3], dtype="uint8")))

    @pytest.mark.skip
    def test_data_structure_as_variables_correctly_allocates_cuda_tensors(self):
        # TODO(Mark): Work out if we can test this somehow without actual GPUs.
        array_dict = {
                "sentence": {
                        "words": numpy.zeros([3, 4], dtype="float32"),
                        "characters": numpy.ones([2, 5], dtype="int32")
                        },
                "tags": numpy.ones([2, 3], dtype="uint8")
        }
        torch_array_dict = arrays_to_variables(array_dict, cuda_device=1)

        assert torch_array_dict["sentence"]["words"].data.equal(
                torch.cuda.FloatTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].data.equal(
                torch.cuda.IntTensor(numpy.ones([2, 5], dtype="int32")))
        assert torch_array_dict["tags"].data.equal(
                torch.cuda.ByteTensor(numpy.ones([2, 3], dtype="uint8")))


import numpy
import torch

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.common.tensor import data_structure_as_tensors

class TestTesor(AllenNlpTestCase):

    def test_data_structure_as_tensors_handles_recursion(self):

        array_dict = {"sentence": {"words": numpy.zeros([3, 4]),
                                   "characters": numpy.ones([2, 5])},
                      "tags": numpy.ones([2, 3])}
        torch_array_dict = data_structure_as_tensors(array_dict)

        assert torch_array_dict["sentence"]["words"].equal(torch.DoubleTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].equal(torch.DoubleTensor(numpy.ones([2, 5])))
        assert torch_array_dict["tags"].equal(torch.DoubleTensor(numpy.ones([2, 3])))
        print(torch_array_dict)

    def test_data_structure_as_tensors_correctly_converts_mixed_types(self):

        array_dict = {"sentence": {"words": numpy.zeros([3, 4], dtype="float32"),
                                   "characters": numpy.ones([2, 5], dtype="int32")},
                      "tags": numpy.ones([2, 3], dtype="uint8")}
        torch_array_dict = data_structure_as_tensors(array_dict)

        assert torch_array_dict["sentence"]["words"].equal(torch.FloatTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].equal(torch.IntTensor(numpy.ones([2, 5], dtype="int32")))
        assert torch_array_dict["tags"].equal(torch.ByteTensor(numpy.ones([2, 3], dtype="uint8")))

    def test_data_structure_as_tensors_correctly_allocates_cuda_tensors(self):

        array_dict = {"sentence": {"words": numpy.zeros([3, 4], dtype="float32"),
                                   "characters": numpy.ones([2, 5], dtype="int32")},
                      "tags": numpy.ones([2, 3], dtype="uint8")}
        torch_array_dict = data_structure_as_tensors(array_dict, cuda_device=1)

        assert torch_array_dict["sentence"]["words"].equal(torch.cuda.FloatTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].equal(torch.cuda.IntTensor(numpy.ones([2, 5], dtype="int32")))
        assert torch_array_dict["tags"].equal(torch.cuda.ByteTensor(numpy.ones([2, 3], dtype="uint8")))
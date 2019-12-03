from numpy.testing import assert_almost_equal
from overrides import overrides
import torch
from torch.nn import Embedding, Module, Parameter

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import TimeDistributed


class TestTimeDistributed(AllenNlpTestCase):
    def test_time_distributed_reshapes_named_arg_correctly(self):
        char_embedding = Embedding(2, 2)
        char_embedding.weight = Parameter(torch.FloatTensor([[0.4, 0.4], [0.5, 0.5]]))
        distributed_embedding = TimeDistributed(char_embedding)
        char_input = torch.LongTensor([[[1, 0], [1, 1]]])
        output = distributed_embedding(char_input)
        assert_almost_equal(
            output.data.numpy(), [[[[0.5, 0.5], [0.4, 0.4]], [[0.5, 0.5], [0.5, 0.5]]]]
        )

    def test_time_distributed_reshapes_positional_kwarg_correctly(self):
        char_embedding = Embedding(2, 2)
        char_embedding.weight = Parameter(torch.FloatTensor([[0.4, 0.4], [0.5, 0.5]]))
        distributed_embedding = TimeDistributed(char_embedding)
        char_input = torch.LongTensor([[[1, 0], [1, 1]]])
        output = distributed_embedding(input=char_input)
        assert_almost_equal(
            output.data.numpy(), [[[[0.5, 0.5], [0.4, 0.4]], [[0.5, 0.5], [0.5, 0.5]]]]
        )

    def test_time_distributed_works_with_multiple_inputs(self):
        module = lambda x, y: x + y
        distributed = TimeDistributed(module)
        x_input = torch.LongTensor([[[1, 2], [3, 4]]])
        y_input = torch.LongTensor([[[4, 2], [9, 1]]])
        output = distributed(x_input, y_input)
        assert_almost_equal(output.data.numpy(), [[[5, 4], [12, 5]]])

    def test_time_distributed_reshapes_multiple_inputs_with_pass_through_tensor_correctly(self):
        class FakeModule(Module):
            @overrides
            def forward(self, input_tensor, tensor_to_pass_through=None, another_tensor=None):

                return input_tensor + tensor_to_pass_through + another_tensor

        module = FakeModule()
        distributed_module = TimeDistributed(module)

        input_tensor1 = torch.LongTensor([[[1, 2], [3, 4]]])
        input_to_pass_through = torch.LongTensor([3, 7])
        input_tensor2 = torch.LongTensor([[[4, 2], [9, 1]]])

        output = distributed_module(
            input_tensor1,
            tensor_to_pass_through=input_to_pass_through,
            another_tensor=input_tensor2,
            pass_through=["tensor_to_pass_through"],
        )
        assert_almost_equal(output.data.numpy(), [[[8, 11], [15, 12]]])

    def test_time_distributed_reshapes_multiple_inputs_with_pass_through_non_tensor_correctly(self):
        class FakeModule(Module):
            @overrides
            def forward(self, input_tensor, number=0, another_tensor=None):

                return input_tensor + number + another_tensor

        module = FakeModule()
        distributed_module = TimeDistributed(module)

        input_tensor1 = torch.LongTensor([[[1, 2], [3, 4]]])
        input_number = 5
        input_tensor2 = torch.LongTensor([[[4, 2], [9, 1]]])

        output = distributed_module(
            input_tensor1,
            number=input_number,
            another_tensor=input_tensor2,
            pass_through=["number"],
        )
        assert_almost_equal(output.data.numpy(), [[[10, 9], [17, 10]]])

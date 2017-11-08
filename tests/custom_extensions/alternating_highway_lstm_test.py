import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
class TestCustomHighwayLSTM(AllenNlpTestCase):

    def test_small_model(self):
        args = self.get_models_and_inputs(5, 3, 11, 2, 5, 0.0)
        self.forward_and_backward_outputs_match(*args)

    def test_large_model(self):
        args = self.get_models_and_inputs(83, 103, 311, 8, 101, 0.0)
        self.forward_and_backward_outputs_match(*args)

    def test_validation_forward_pass_is_deterministic_in_model_with_dropout(self):

        _, model, _, model_input, lengths = self.get_models_and_inputs(5, 3, 11, 2, 5, dropout_prob=0.5)
        model.eval()
        model_input = pack_padded_sequence(model_input, lengths, batch_first=True)
        output, _ = model(model_input)
        output, _ = pad_packed_sequence(output, batch_first=True)

        for i in range(3):
            output_new, _ = model(model_input)
            output_new, _ = pad_packed_sequence(output_new, batch_first=True)
            numpy.testing.assert_array_almost_equal(output.data.cpu().numpy(), output_new.data.cpu().numpy())
            output = output_new

    @staticmethod
    def forward_and_backward_outputs_match(baseline_model, kernel_model,
                                           baseline_input, kernel_input, lengths):

        packed_baseline_input = pack_padded_sequence(baseline_input, lengths, batch_first=True)
        baseline_output, _ = baseline_model(packed_baseline_input)
        baseline_output, _ = pad_packed_sequence(baseline_output, batch_first=True)

        packed_kernel_input = pack_padded_sequence(kernel_input, lengths, batch_first=True)
        kernel_output, _ = kernel_model(packed_kernel_input)
        kernel_output, _ = pad_packed_sequence(kernel_output, batch_first=True)

        numpy.testing.assert_array_almost_equal(baseline_output.data.cpu().numpy(),
                                                kernel_output.data.cpu().numpy())

        # Backprop some random error.
        random_error = torch.randn(baseline_output.size()).cuda()
        baseline_model.zero_grad()
        baseline_output.backward(random_error)

        kernel_model.zero_grad()
        kernel_output.backward(random_error)
        
        numpy.testing.assert_array_almost_equal(baseline_input.grad.data.cpu().numpy(),
                                                kernel_input.grad.data.cpu().numpy())
        weight_index = 0
        bias_index = 0
        for layer in range(baseline_model.num_layers):
            input_grad = getattr(baseline_model, 'layer_%d' % layer).input_linearity.weight.grad
            state_grad = getattr(baseline_model, 'layer_%d' % layer).state_linearity.weight.grad
            bias_grad = getattr(baseline_model, 'layer_%d' % layer).state_linearity.bias.grad

            kernel_input_grad = kernel_model.weight.grad[weight_index: weight_index+input_grad.nelement()]\
                .view(input_grad.size(1), input_grad.size(0)).t()
            weight_index += input_grad.nelement()

            kernel_state_grad = kernel_model.weight.grad[weight_index: weight_index + state_grad.nelement()]\
                .view(state_grad.size(1), state_grad.size(0)).t()
            weight_index += state_grad.nelement()

            kernel_bias_grad = kernel_model.bias.grad[bias_index:bias_index+bias_grad.nelement()]
            bias_index += bias_grad.nelement()

            numpy.testing.assert_array_almost_equal(kernel_input_grad.data.cpu().numpy(),
                                                    input_grad.data.cpu().numpy(), decimal=4) 
            numpy.testing.assert_array_almost_equal(kernel_state_grad.data.cpu().numpy(),
                                                    state_grad.data.cpu().numpy(), decimal=4)
            numpy.testing.assert_array_almost_equal(kernel_bias_grad.data.cpu().numpy(),
                                                    bias_grad.data.cpu().numpy(), decimal=4)

    @staticmethod
    def get_models_and_inputs(batch_size, input_size, output_size, num_layers, timesteps, dropout_prob):

        # Import is here because the layer requires a GPU.
        from allennlp.modules.alternating_highway_lstm import AlternatingHighwayLSTM

        baseline = StackedAlternatingLstm(input_size, output_size, num_layers,
                                          dropout_prob, use_input_projection_bias=False).cuda()
        kernel_version = AlternatingHighwayLSTM(input_size, output_size, num_layers, dropout_prob).cuda()

        # Copy weights from non-cuda version into cuda version,
        # so we are starting from exactly the same place.
        weight_index = 0
        bias_index = 0
        for layer_index in range(num_layers):

            layer = getattr(baseline, 'layer_%d' % layer_index)
            input_weight = layer.input_linearity.weight
            state_weight = layer.state_linearity.weight
            bias = layer.state_linearity.bias

            kernel_version.weight.data[weight_index: weight_index + input_weight.nelement()]\
                .view_as(input_weight.t()).copy_(input_weight.data.t())
            weight_index += input_weight.nelement()

            kernel_version.weight.data[weight_index: weight_index + state_weight.nelement()]\
                .view_as(state_weight.t()).copy_(state_weight.data.t())
            weight_index += state_weight.nelement()

            kernel_version.bias.data[bias_index:bias_index + bias.nelement()].copy_(bias.data)
            bias_index += bias.nelement()

        inputs = torch.randn(batch_size, timesteps, input_size).cuda()
        # Clone variable so different models are
        # completely separate in the graph.
        input2 = inputs.clone()
        baseline_input = Variable(inputs, requires_grad=True)
        kernel_version_input = Variable(input2, requires_grad=True)
        lengths = [timesteps - int((i / 2)) for i in range(batch_size)]
        lengths = lengths[:batch_size]

        return baseline, kernel_version, baseline_input, kernel_version_input, lengths

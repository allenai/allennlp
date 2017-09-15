import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.common.testing import AllenNlpTestCase
from allennlp.custom_extensions.baseline.lstm import AlternatingLSTM
from allennlp.custom_extensions.highway_lstm_layer import HighwayLSTMLayer
from allennlp.custom_extensions.baseline.measurements import Timer


class TestCustomHighwayLSTM(AllenNlpTestCase):

    def test_small_model(self):
        args = self.get_models_and_inputs(5, 3, 11, 2, 5, 0.0)
        self.forward_and_backward_outputs_match(*args)

    def test_small_model_with_dropout(self):
        args = self.get_models_and_inputs(5, 3, 11, 2, 5, 0.5)
        self.forward_and_backward_outputs_match(*args)

    def test_large_model(self):
        args = self.get_models_and_inputs(83, 103, 311, 8, 101, 0.0)
        self.forward_and_backward_outputs_match(*args)

    def test_large_model_with_dropout(self):
        args = self.get_models_and_inputs(83, 103, 311, 8, 101, 0.0)
        self.forward_and_backward_outputs_match(*args)

    @staticmethod
    def forward_and_backward_outputs_match(baseline_model, kernel_model,
                                           baseline_input, kernel_input, dropout):

        with Timer('Baseline'):
            baseline_output = baseline_model(baseline_input, dropout_weights=dropout)
            baseline_output, _ = pad_packed_sequence(baseline_output, batch_first=True)
            print("baseline output: ", baseline_output)

        with Timer("Mine"):
            baseline_output, _ = baseline_input(baseline_input, dropout_weights=dropout)
            baseline_output, _ = pad_packed_sequence(baseline_output, batch_first=True)
            print("CUDA output: ", baseline_output)

        diff = torch.max(baseline_output.data - baseline_output.data)
        assert diff < 1e-4, "Output does not match: " + str(diff)

        # Backprop some random error.
        back_err = torch.randn(baseline_input.size()).cuda()
        baseline_model.zero_grad()
        baseline_output.backward(back_err)

        kernel_model.zero_grad()
        baseline_output.backward(back_err)

        input_grad_diff = torch.max(baseline_input.grad.data - kernel_input.grad.data)
        assert input_grad_diff < 1e-4, "Input grad does not match: " + str(input_grad_diff)

        weight_ind = 0
        bias_ind = 0
        for layer in range(baseline_model.num_layers):
            print("TEST %d" % (layer))
            x_grad = getattr(baseline_model, 'layer_%d' % layer).xlin.weight.grad
            h_grad = getattr(baseline_model, 'layer_%d' % layer).hlin.weight.grad
            bias = getattr(baseline_model, 'layer_%d' % layer).hlin.bias.grad

            mine_x_grad = kernel_model.weight.grad[weight_ind:weight_ind+x_grad.nelement()].view(x_grad.size(1), x_grad.size(0)).t()
            weight_ind += x_grad.nelement()

            mine_h_grad = kernel_model.weight.grad[weight_ind:weight_ind+h_grad.nelement()].view(h_grad.size(1), h_grad.size(0)).t()
            weight_ind += h_grad.nelement()

            mine_bias = kernel_model.bias.grad[bias_ind:bias_ind+bias.nelement()]
            bias_ind += bias.nelement()

            x_diff = torch.max(mine_x_grad.data - x_grad.data)
            assert x_diff < 1e-4, "Layer %d x_weight does not match: " % layer + str(x_diff)

            h_diff = torch.max(mine_h_grad.data - h_grad.data)
            assert h_diff < 1e-4, "Layer %d h_weight does not match: " % layer + str(h_diff)

            bias_diff = torch.max(mine_bias.data - bias.data)
            assert bias_diff < 1e-4, "Layer %d bias does not match: " % layer + str(bias_diff)

    def get_models_and_inputs(self, batch_size, input_size,
                              output_size, num_layers, timesteps, dropout_prob):

        baseline = AlternatingLSTM(input_size,
                                   output_size,
                                   num_layers,
                                   recurrent_dropout_prob=dropout_prob).cuda()
        kernel_version = HighwayLSTMLayer(input_size,
                                          output_size,
                                          num_layers=num_layers,
                                          recurrent_dropout_prob=dropout_prob).cuda()
        curr_weight_ind = 0
        curr_bias_ind = 0
        print("CUDA lstm - weight elements: ", kernel_version.weight.nelement())
        for layer in range(num_layers):
            print("Layer: ", layer, "Weight Index: ", curr_weight_ind)
            print("bias index: ", curr_bias_ind)
            x_weight = getattr(baseline, 'layer_%d' % layer).xlin.weight
            h_weight = getattr(baseline, 'layer_%d' % layer).hlin.weight
            bias = getattr(baseline, 'layer_%d' % layer).hlin.bias
            kernel_version.weight.data[curr_weight_ind:curr_weight_ind+x_weight.nelement()].copy_(x_weight.data.t())
            curr_weight_ind += x_weight.nelement()
            kernel_version.weight.data[curr_weight_ind:curr_weight_ind+h_weight.nelement()].copy_(h_weight.data.t())
            curr_weight_ind += h_weight.nelement()
            kernel_version.bias.data[curr_bias_ind:curr_bias_ind+bias.nelement()].copy_(bias.data)
            curr_bias_ind += bias.nelement()

        input = torch.randn(batch_size, timesteps, input_size).cuda()
        input2 = input.clone()
        baseline_input = Variable(input, requires_grad=True)
        mine_input = Variable(input2, requires_grad=True)
        lengths = [timesteps - (i / 2) for i in range(batch_size)]
        lengths = lengths[:batch_size]
        baseline_input = pack_padded_sequence(baseline_input, lengths, batch_first=True)
        kernel_version_input = pack_padded_sequence(mine_input, lengths, batch_first=True)
        if dropout_prob > 0:
            dropout = Variable(torch.Tensor(num_layers,
                                            batch_size,
                                            output_size).cuda().bernoulli_(dropout_prob))
        else:
            dropout = None

        return baseline, kernel_version, baseline_input, kernel_version_input, dropout

from flaky import flaky
import pytest
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.weight_drop import WeightDropout




class WeightDropoutTest(AllenNlpTestCase):

    @flaky(max_runs=10, min_passes=1)
    def test_linear_outputs(self):
        # Check that weights are (probably) being dropped out properly. There's an extremely small
        # chance (p < 1e-86) that this test fails.
        x = torch.ones(10, dtype=torch.float32)
        weight_dropped_linear = WeightDropout(torch.nn.Linear(10, 10),
                                              parameter_regex='weight',
                                              dropout=0.9)

        # Check that outputs differ if module is in training mode
        weight_dropped_linear.train()
        output_a = weight_dropped_linear(x)
        output_b = weight_dropped_linear(x)
        assert not torch.allclose(output_a, output_b)

        # Check that outputs are the same if module is in eval mode
        weight_dropped_linear.eval()
        output_a = weight_dropped_linear(x)
        output_b = weight_dropped_linear(x)
        assert torch.allclose(output_a, output_b)

    @flaky(max_runs=10, min_passes=1)
    def test_lstm_outputs(self):
        # Check that lstm weights are (probably) being dropped out properly. There's an extremely
        # small chance (p < 1e-86) that this test fails.
        x = torch.ones(1, 2, 10, dtype=torch.float32)  # shape: (batch, seq_length, dim)
        lstm = torch.nn.LSTM(input_size=10, hidden_size=10, batch_first=True)
        weight_dropped_lstm = WeightDropout(module=lstm, parameter_regex='weight_hh', dropout=0.9)

        # Check that outputs differ if module is in training mode. Since only hidden-to-hidden
        # weights are masked, the first outputs should be the same.
        weight_dropped_lstm.train()
        output_a, _ = weight_dropped_lstm(x)
        output_b, _ = weight_dropped_lstm(x)
        assert torch.allclose(output_a[:, 0, :], output_b[:, 0, :])
        assert not torch.allclose(output_a[:, 1, :], output_b[:, 1, :])

        # Check that outputs are the same if module is in eval mode
        weight_dropped_lstm.eval()
        output_a, _ = weight_dropped_lstm(x)
        output_b, _ = weight_dropped_lstm(x)
        assert torch.allclose(output_a, output_b)

    def test_all_parameters_matched_and_moved_to_top_level_module(self):
        # Check that regex properly matches parameters in all submodules, and that the matched
        # parameters are properly moved to the top-level WeightDrop module.

        # Create a mock module which contains a linear layer as well as a parameter in the top-level module.
        class Mock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
                # Use identical names to linear params to test that we don't run into name conflicts.
                self.weight = torch.nn.Parameter(torch.tensor([1., 2]))
                self.bias = torch.nn.Parameter(torch.tensor([1., 2]))

        module = Mock()
        weight_dropped_module = WeightDropout(module=module, parameter_regex='weight', dropout=0.9)

        # We are ensuring the following:
        #   1. That there is a '_raw' version of the matched parameters associated to the top-level module.
        #   2. That the original matched parameters are deleted from their respective parent modules.
        #   3. That nothing has happened to the unnmatched parameters.
        parameter_names = {name for name, _ in weight_dropped_module.named_parameters()}
        expected_parameter_names = {'_module_weight_raw', '_module.bias', '_module_linear_weight_raw', '_module.linear.bias'}
        self.assertSetEqual(parameter_names, expected_parameter_names)

    def test_parameters_are_leaf_tensors(self):
        # Checks that WeightDrop parameters are always leaf tensors.

        # Case 1: After initialization
        weight_dropped_linear = WeightDropout(torch.nn.Linear(10, 10),
                                              parameter_regex='weight',
                                              dropout=0.9)
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 2: When in training mode
        weight_dropped_linear.train()
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 3: After forward
        x = torch.ones(10, dtype=torch.float32)
        weight_dropped_linear(x)
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 4: After reset()
        weight_dropped_linear.reset()
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 5: When in eval mode
        weight_dropped_linear.eval()
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())




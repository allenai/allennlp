from flaky import flaky
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.drop_connect import DropConnect


class DropConnectTest(AllenNlpTestCase):
    @flaky(max_runs=10, min_passes=1)
    def test_linear_outputs(self):
        # Check that weights are (probably) being dropped out properly. There's an extremely small
        # chance (p < 1e-86) that this test fails.
        input_tensor = torch.ones(10, dtype=torch.float32)
        dropped_linear = DropConnect(torch.nn.Linear(10, 10), parameter_regex="weight", dropout=0.9)

        # Check that outputs differ if module is in training mode
        dropped_linear.train()
        output_a = dropped_linear(input_tensor)
        output_b = dropped_linear(input_tensor)
        assert not torch.allclose(output_a, output_b)

        # Check that outputs are the same if module is in eval mode
        dropped_linear.eval()
        output_a = dropped_linear(input_tensor)
        output_b = dropped_linear(input_tensor)
        assert torch.allclose(output_a, output_b)

    @flaky(max_runs=10, min_passes=1)
    def test_lstm_outputs(self):
        # Check that lstm weights are (probably) being dropped out properly. There's an extremely
        # small chance (p < 1e-86) that this test fails.
        input_tensor = torch.ones(1, 2, 10, dtype=torch.float32)  # shape: (batch, seq_length, dim)
        lstm = torch.nn.LSTM(input_size=10, hidden_size=10, batch_first=True)
        dropped_lstm = DropConnect(module=lstm, parameter_regex="weight_hh", dropout=0.9)

        # Check that outputs differ if module is in training mode. Since only hidden-to-hidden
        # weights are masked, the first outputs should be the same.
        dropped_lstm.train()
        output_a, _ = dropped_lstm(input_tensor)
        output_b, _ = dropped_lstm(input_tensor)
        assert torch.allclose(output_a[:, 0, :], output_b[:, 0, :])
        assert not torch.allclose(output_a[:, 1, :], output_b[:, 1, :])

        # Check that outputs are the same if module is in eval mode
        dropped_lstm.eval()
        output_a, _ = dropped_lstm(input_tensor)
        output_b, _ = dropped_lstm(input_tensor)
        assert torch.allclose(output_a, output_b)

    def test_matched_params_correctly_moved(self):
        # Check that regex properly matches parameters in all submodules, and that the matched
        # parameters are properly moved to the top-level DropConnect module.

        # Create a mock module which contains a linear layer as well as a parameter in the top-level module.
        class Mock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
                # Use identical names to linear params to test that we don't run into name conflicts.
                self.weight = torch.nn.Parameter(torch.FloatTensor([1, 2]))
                self.bias = torch.nn.Parameter(torch.FloatTensor([1.0, 2]))

        module = Mock()
        dropped_module = DropConnect(module=module, parameter_regex="weight", dropout=0.9)

        # We are ensuring the following:
        #   1. That there is a '_raw' version of the matched parameters associated to the top-level module.
        #   2. That the original matched parameters are deleted from their respective parent modules.
        #   3. That nothing has happened to the unnmatched parameters.
        parameter_names = {name for name, _ in dropped_module.named_parameters()}
        expected_parameter_names = {
            "_module_weight_raw",
            "_module.bias",
            "_module_linear_weight_raw",
            "_module.linear.bias",
        }
        self.assertSetEqual(parameter_names, expected_parameter_names)

    def test_parameters_are_leaf_tensors(self):
        # Checks that WeightDrop parameters are always leaf tensors.

        # Case 1: After initialization
        weight_dropped_linear = DropConnect(
            torch.nn.Linear(10, 10), parameter_regex="weight", dropout=0.9
        )
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 2: When in training mode
        weight_dropped_linear.train()
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 3: After forward
        input_tensor = torch.ones(10, dtype=torch.float32)
        weight_dropped_linear(input_tensor)
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 4: After reset()
        weight_dropped_linear.reset()
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

        # Case 5: When in eval mode
        weight_dropped_linear.eval()
        assert all(parameter.is_leaf for parameter in weight_dropped_linear.parameters())

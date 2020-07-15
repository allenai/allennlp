from typing import Dict, Optional

import pytest
import torch

from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.initializers import PretrainedModelInitializer
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params


class _Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_2 = torch.nn.Linear(10, 5)
        self.scalar = torch.nn.Parameter(torch.rand(()))

    def forward(self, inputs):
        pass


class _Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_3 = torch.nn.Linear(10, 5)
        self.scalar = torch.nn.Parameter(torch.rand(()))

    def forward(self, inputs):
        pass


class TestPretrainedModelInitializer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.net1 = _Net1()
        self.net2 = _Net2()
        self.temp_file = self.TEST_DIR / "weights.th"
        torch.save(self.net2.state_dict(), self.temp_file)

    def _are_equal(self, linear1: torch.nn.Linear, linear2: torch.nn.Linear) -> bool:
        return torch.equal(linear1.weight, linear2.weight) and torch.equal(
            linear1.bias, linear2.bias
        )

    def _get_applicator(
        self,
        regex: str,
        weights_file_path: str,
        parameter_name_overrides: Optional[Dict[str, str]] = None,
    ) -> InitializerApplicator:
        initializer = PretrainedModelInitializer(weights_file_path, parameter_name_overrides)
        return InitializerApplicator([(regex, initializer)])

    def test_random_initialization(self):
        # The tests in the class rely on the fact that the parameters for
        # `self.net1` and `self.net2` are randomly initialized and not
        # equal at the beginning. This test makes sure that's true
        assert not self._are_equal(self.net1.linear_1, self.net2.linear_1)
        assert not self._are_equal(self.net1.linear_2, self.net2.linear_3)

    def test_from_params(self):
        params = Params({"type": "pretrained", "weights_file_path": self.temp_file})
        initializer = Initializer.from_params(params)
        assert initializer.weights
        assert initializer.parameter_name_overrides == {}

        name_overrides = {"a": "b", "c": "d"}
        params = Params(
            {
                "type": "pretrained",
                "weights_file_path": self.temp_file,
                "parameter_name_overrides": name_overrides,
            }
        )
        initializer = Initializer.from_params(params)
        assert initializer.weights
        assert initializer.parameter_name_overrides == name_overrides

    def test_default_parameter_names(self):
        # This test initializes net1 to net2's parameters. It doesn't use
        # the parameter name overrides, so it will verify the initialization
        # works if the two parameters' names are the same.
        applicator = self._get_applicator("linear_1.weight|linear_1.bias", self.temp_file)
        applicator(self.net1)
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1)
        assert not self._are_equal(self.net1.linear_2, self.net2.linear_3)

    def test_parameter_name_overrides(self):
        # This test will use the parameter name overrides to initialize all
        # of net1's weights to net2's.
        name_overrides = {"linear_2.weight": "linear_3.weight", "linear_2.bias": "linear_3.bias"}
        applicator = self._get_applicator("linear_*", self.temp_file, name_overrides)
        applicator(self.net1)
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1)
        assert self._are_equal(self.net1.linear_2, self.net2.linear_3)

    def test_size_mismatch(self):
        # This test will verify that an exception is raised when you try
        # to initialize a parameter to a pretrained parameter and they have
        # different sizes
        name_overrides = {"linear_1.weight": "linear_3.weight"}
        applicator = self._get_applicator("linear_1.*", self.temp_file, name_overrides)
        with pytest.raises(ConfigurationError):
            applicator(self.net1)

    def test_zero_dim_tensor(self):
        # This test will verify that a 0-dim tensor can be initialized.
        # It raises IndexError if slicing a tensor to copy the parameter.
        applicator = self._get_applicator("scalar", self.temp_file)
        applicator(self.net1)
        assert torch.equal(self.net1.scalar, self.net2.scalar)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_load_to_gpu_from_gpu(self):
        # This test will make sure that the initializer works on the GPU
        self.net1.cuda(device=0)
        self.net2.cuda(device=0)

        # Verify the parameters are on the GPU
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True
        assert self.net2.linear_1.weight.is_cuda is True
        assert self.net2.linear_1.bias.is_cuda is True

        # We need to manually save the parameters to a file because setup_method()
        # only does it for the CPU
        temp_file = self.TEST_DIR / "gpu_weights.th"
        torch.save(self.net2.state_dict(), temp_file)

        applicator = self._get_applicator("linear_1.*", temp_file)
        applicator(self.net1)

        # Verify the parameters are still on the GPU
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True
        assert self.net2.linear_1.weight.is_cuda is True
        assert self.net2.linear_1.bias.is_cuda is True

        # Make sure the weights are identical
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_load_to_cpu_from_gpu(self):
        # This test will load net2's parameters onto the GPU, then use them to
        # initialize net1 on the CPU
        self.net2.cuda(device=0)

        # Verify the parameters are on the GPU
        assert self.net2.linear_1.weight.is_cuda is True
        assert self.net2.linear_1.bias.is_cuda is True

        temp_file = self.TEST_DIR / "gpu_weights.th"
        torch.save(self.net2.state_dict(), temp_file)

        applicator = self._get_applicator("linear_1.*", temp_file)
        applicator(self.net1)

        # Verify the parameters are on the CPU
        assert self.net1.linear_1.weight.is_cuda is False
        assert self.net1.linear_1.bias.is_cuda is False

        # Make sure the weights are identical
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_load_to_gpu_from_cpu(self):
        # This test will load net1's parameters onto the GPU, then use net2's
        # on the CPU to initialize net1's parameters.
        self.net1.cuda(device=0)

        # Verify the parameters are on the GPU
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True

        # net2's parameters are already saved to CPU from setup_method()
        applicator = self._get_applicator("linear_1.*", self.temp_file)
        applicator(self.net1)

        # Verify the parameters are on the GPU
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True

        # Make sure the weights are identical
        assert self._are_equal(self.net1.linear_1.cpu(), self.net2.linear_1)

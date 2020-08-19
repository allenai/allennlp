from numpy.testing import assert_almost_equal
import inspect
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator, Initializer, Activation
from allennlp.common.testing import AllenNlpTestCase


class TestFeedForward(AllenNlpTestCase):
    def test_can_construct_from_params(self):
        params = Params({"input_dim": 2, "hidden_dims": 3, "activations": "relu", "num_layers": 2})
        feedforward = FeedForward.from_params(params)
        assert len(feedforward._activations) == 2
        assert [isinstance(a, torch.nn.ReLU) for a in feedforward._activations]
        assert len(feedforward._linear_layers) == 2
        assert [layer.weight.size(-1) == 3 for layer in feedforward._linear_layers]

        params = Params(
            {
                "input_dim": 2,
                "hidden_dims": [3, 4, 5],
                "activations": ["relu", "relu", "linear"],
                "dropout": 0.2,
                "num_layers": 3,
            }
        )
        feedforward = FeedForward.from_params(params)
        assert len(feedforward._activations) == 3
        assert isinstance(feedforward._activations[0], torch.nn.ReLU)
        assert isinstance(feedforward._activations[1], torch.nn.ReLU)
        # It's hard to check that the last activation is the lambda function we use for `linear`,
        # so this is good enough.
        assert not isinstance(feedforward._activations[2], torch.nn.ReLU)

        assert len(feedforward._linear_layers) == 3
        assert feedforward._linear_layers[0].weight.size(0) == 3
        assert feedforward._linear_layers[1].weight.size(0) == 4
        assert feedforward._linear_layers[2].weight.size(0) == 5

        assert len(feedforward._dropout) == 3
        assert [d.p == 0.2 for d in feedforward._dropout]

    def test_init_checks_hidden_dim_consistency(self):
        with pytest.raises(ConfigurationError):
            FeedForward(2, 4, [5, 5], Activation.by_name("relu")())

    def test_init_checks_activation_consistency(self):
        with pytest.raises(ConfigurationError):
            FeedForward(2, 4, 5, [Activation.by_name("relu")(), Activation.by_name("relu")()])

    def test_forward_gives_correct_output(self):
        params = Params({"input_dim": 2, "hidden_dims": 3, "activations": "relu", "num_layers": 2})
        feedforward = FeedForward.from_params(params)

        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.0}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(feedforward)

        input_tensor = torch.FloatTensor([[-3, 1]])
        output = feedforward(input_tensor).data.numpy()
        assert output.shape == (1, 3)
        # This output was checked by hand - ReLU makes output after first hidden layer [0, 0, 0],
        # which then gets a bias added in the second layer to be [1, 1, 1].
        assert_almost_equal(output, [[1, 1, 1]])

    def test_textual_representation_contains_activations(self):
        params = Params(
            {
                "input_dim": 2,
                "hidden_dims": 3,
                "activations": ["linear", "relu", "swish"],
                "num_layers": 3,
            }
        )
        feedforward = FeedForward.from_params(params)
        expected_text_representation = inspect.cleandoc(
            """
            FeedForward(
              (_activations): ModuleList(
                (0): Linear()
                (1): ReLU()
                (2): Swish()
              )
              (_linear_layers): ModuleList(
                (0): Linear(in_features=2, out_features=3, bias=True)
                (1): Linear(in_features=3, out_features=3, bias=True)
                (2): Linear(in_features=3, out_features=3, bias=True)
              )
              (_dropout): ModuleList(
                (0): Dropout(p=0.0, inplace=False)
                (1): Dropout(p=0.0, inplace=False)
                (2): Dropout(p=0.0, inplace=False)
              )
            )
            """
        )
        actual_text_representation = str(feedforward)

        assert actual_text_representation == expected_text_representation

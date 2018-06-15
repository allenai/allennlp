# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.modules.softmax_with_nlls import AdaptiveSoftmax
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data import Vocabulary


class TestAdaptiveSoftmax(AllenNlpTestCase):


    def test_from_params_builders_softmax_correctly(self):

        vocab0 = Vocabulary(counter={"tokens": {"hello": 1, "world": 2, ":": 3, "-": 4}})
        params0 = Params({
                "input_dim": 10,
                "cutoff": [1, 3],
                "label_namespace": "tokens"
                })
        softmax = AdaptiveSoftmax.from_params(vocab0, params0)

        assert softmax.input_dim == 10
        assert softmax.cutoff == [1, 3, 6]
        assert softmax.output_size == 3
        assert softmax.adaptive == True
        assert len(softmax.tail) == 2

        vocab0 = Vocabulary(counter={"tokens": {"hello": 1, "world": 2, ":": 3, "-": 4}})
        params0 = Params({
                "input_dim": 10,
                "cutoff": [],
                "label_namespace": "tokens"
                })
        softmax = AdaptiveSoftmax.from_params(vocab0, params0)

        assert softmax.input_dim == 10
        assert softmax.cutoff == [6]
        assert softmax.output_size == 6
        assert softmax.adaptive == False
        assert len(softmax.tail) == 0


    def test_from_params_requires_cutoff_mono_increasing(self):

        vocab0 = Vocabulary(counter={"tokens": {"hello": 1, "world": 2, ":": 3, "-": 4}})
        params0 = Params({
                "input_dim": 10,
                "cutoff": [3, 1],
                "label_namespace": "tokens"
                })
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            softmax = AdaptiveSoftmax.from_params(vocab0, params0)

        vocab1 = Vocabulary(counter={"tokens": {"hello": 1, "world": 2, ":": 3, "-": 4}})
        params1 = Params({
                "input_dim": 10,
                "cutoff": [1, 3],
                "label_namespace": "tokens"
                })

        softmax = AdaptiveSoftmax.from_params(vocab1, params1)


    def test_from_params_requires_cutoff_larger_than_vocab_size(self):

        vocab0 = Vocabulary(counter={"tokens": {"hello": 1}})
        params0 = Params({
                "input_dim": 10,
                "cutoff": [3],
                "label_namespace": "tokens"
                })
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            softmax = AdaptiveSoftmax.from_params(vocab0, params0)

        vocab1 = Vocabulary(counter={"tokens": {"hello": 1, "world": 2}})
        params1 = Params({
                "input_dim": 10,
                "cutoff": [3],
                "label_namespace": "tokens"
                })
        softmax = AdaptiveSoftmax.from_params(vocab1, params1)
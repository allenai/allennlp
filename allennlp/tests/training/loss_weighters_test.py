# pylint: disable=no-self-use,invalid-name,protected-access

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.loss_weighters import LossWeighter
from allennlp.common.params import Params

class LossWeightersTest(AllenNlpTestCase):

    def test_constant_weighter(self):
        initial_weight = 1.0
        steps = 10
        constant_weight = Params({"weighter": {"type": "constant_weight", "initial_weight": initial_weight}})
        loss_weights = LossWeighter.from_params(constant_weight)
        assert [loss_weights["weighter"].next() for _ in range(steps)] == [initial_weight] * steps

    def test_no_warmup_no_min_linear_annealer(self):
        num_iter_to_max = 10
        linear_annealer = Params({"weighter": {"type": "linear_annealer", "min_weight": 0,
                                               "max_weight": 10, "warmup": 0, "num_iter_to_max": num_iter_to_max}})
        loss_weights = LossWeighter.from_params(linear_annealer)
        assert loss_weights["weighter"].get() == 0.0
        assert [loss_weights["weighter"].next() for _ in range(num_iter_to_max + 1)] == [0.0, 1.0, 2.0, 3.0,
                                                                                         4.0, 5.0, 6.0, 7.0,
                                                                                         8.0, 9.0, 10.0]
        assert loss_weights["weighter"].get() == 10.0

    def test_no_warmup_linear_annealer(self):
        num_iter_to_max = 10
        linear_annealer = Params({"weighter": {"type": "linear_annealer", "min_weight": 10,
                                               "max_weight": 20, "warmup": 0, "num_iter_to_max": num_iter_to_max}})
        loss_weights = LossWeighter.from_params(linear_annealer)
        assert loss_weights["weighter"].get() == 10.0
        assert [loss_weights["weighter"].next() for _ in range(num_iter_to_max + 1)] == [10.0, 11.0, 12.0, 13.0,
                                                                                         14.0, 15.0, 16.0, 17.0,
                                                                                         18.0, 19.0, 20.0]
        assert loss_weights["weighter"].get() == 20.0

    def test_linear_annealer(self):
        num_iter_to_max = 17
        linear_annealer = Params({"weighter": {"type": "linear_annealer", "min_weight": 10,
                                               "max_weight": 20, "warmup": 7, "num_iter_to_max": num_iter_to_max}})
        loss_weights = LossWeighter.from_params(linear_annealer)
        assert loss_weights["weighter"].get() == 10.0
        assert [loss_weights["weighter"].next() for _ in range(num_iter_to_max + 1)] == \
                                                                    [10.0, 10.0, 10.0, 10.0, 10.0,
                                                                    10.0, 10.0, 10.0, 11.0, 12.0,
                                                                    13.0, 14.0, 15.0, 16.0, 17.0,
                                                                    18.0, 19.0, 20.0]
        assert loss_weights["weighter"].get() == 20.0

    def test_no_warmup_no_min_sigmoid_annealer(self):
        num_iter_to_max = 10
        sigmoid_annealer = Params({"weighter": {"type": "sigmoid_annealer", "min_weight": 0,
                                                "max_weight": 10, "warmup": 0,
                                                "num_iter_to_max": num_iter_to_max, "slope": 1}})
        loss_weights = LossWeighter.from_params(sigmoid_annealer)
        assert loss_weights["weighter"].get() == 0.0
        assert [loss_weights["weighter"].next() for _ in range(num_iter_to_max + 1)] == [0.07, 0.18, 0.47, 1.19,
                                                                                         2.69, 5.0, 7.31, 8.81,
                                                                                         9.53, 9.82, 9.93]
        assert loss_weights["weighter"].get() == 9.93

    def test_no_warmup_sigmoid_annealer(self):
        num_iter_to_max = 10
        sigmoid_annealer = Params({"weighter": {"type": "sigmoid_annealer", "min_weight": 10,
                                                "max_weight": 20, "warmup": 0, 
                                                "num_iter_to_max": num_iter_to_max, "slope": 1}})
        loss_weights = LossWeighter.from_params(sigmoid_annealer)
        assert loss_weights["weighter"].get() == 10.0
        assert [loss_weights["weighter"].next() for _ in range(num_iter_to_max + 1)] ==  \
                                                                    [10.07, 10.18, 10.47, 11.19,
                                                                     12.69, 15.0, 17.31, 18.81,
                                                                     19.53, 19.82, 19.93]
        assert loss_weights["weighter"].get() == 19.93

    def test_sigmoid_annealer(self):
        num_iter_to_max = 17
        sigmoid_annealer = Params({"weighter": {"type": "sigmoid_annealer", "min_weight": 10,
                                                "max_weight": 20, "warmup": 7,
                                                "num_iter_to_max": num_iter_to_max, "slope": 1}})
        loss_weights = LossWeighter.from_params(sigmoid_annealer)
        assert loss_weights["weighter"].get() == 10.0
        assert [loss_weights["weighter"].next() for _ in range(num_iter_to_max + 1)] == \
                                                                    [10.0, 10.0, 10.0, 10.0, 10.0,
                                                                     10.01, 10.02, 10.07, 10.18,
                                                                     10.47, 11.19, 12.69, 15.0, 17.31,
                                                                     18.81, 19.53, 19.82, 19.93]
        assert loss_weights["weighter"].get() == 19.93

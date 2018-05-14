# pylint: disable=no-self-use,invalid-name,protected-access

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common.params import Params


class LearningRateSchedulersTest(AllenNlpTestCase):

    def test_reduce_on_plateau_error_throw_when_no_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        with self.assertRaises(ConfigurationError) as context:
            LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                    Params({"type": "adam"})),
                                              Params({"type": "reduce_on_plateau"})).step(None, None)

        self.assertTrue(
            'The reduce_on_plateau learning rate scheduler requires a validation metric' in str(context.exception))

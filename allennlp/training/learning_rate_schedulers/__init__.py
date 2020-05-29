"""
AllenNLP uses most
`PyTorch learning rate schedulers <https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_,
with a thin wrapper to allow registering them and instantiating them `from_params`.

The available learning rate schedulers from PyTorch are

* `"step" <https://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
* `"multi_step" <https://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
* `"exponential" <https://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
* `"reduce_on_plateau" <https://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_

In addition, AllenNLP also provides `cosine with restarts <https://arxiv.org/abs/1608.03983>`_,
a Noam schedule, and a slanted triangular schedule, which are registered as
"cosine", "noam", and "slanted_triangular", respectively.
"""

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import (
    LearningRateScheduler,
    StepLearningRateScheduler,
    MultiStepLearningRateScheduler,
    ExponentialLearningRateScheduler,
    ReduceOnPlateauLearningRateScheduler,
)
from allennlp.training.learning_rate_schedulers.cosine import CosineWithRestarts
from allennlp.training.learning_rate_schedulers.noam import NoamLR
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
from allennlp.training.learning_rate_schedulers.polynomial_decay import PolynomialDecay

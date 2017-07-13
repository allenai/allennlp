from allennlp.common import Params
from allennlp.experiments import drivers


class Driver:
    """
    A ``Driver`` is an entry point into running some experiment with AllenNLP.  For example, there
    is a ``TrainDriver`` for training a model given some data, and a ``TestDriver`` for evaluating
    a trained model a new dataset.

    ``Driver`` just defines one method: ``run()``.  After instantiating the object, you call
    ``driver.run()`` to actually run the operation defined by the driver.
    """
    def run(self):
        """
        Runs the operation defined by the ``Driver``.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        # TODO(mattg)
        operation = params.pop_choice("operation", list(drivers.keys()), default_to_first_choice=True)
        return drivers[operation].from_params(params)

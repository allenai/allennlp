from allennlp.common import Params


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
        from allennlp.experiments.registry import Registry

        operation = params.pop_choice("operation", Registry.list_drivers(), default_to_first_choice=True)

        return Registry.get_driver(operation).from_params(params)

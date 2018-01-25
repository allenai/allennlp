from typing import Iterable

from allennlp.data.instance import Instance, InstanceGenerator
from allennlp.common import Params
from allennlp.common.registrable import Registrable
from allennlp.common.util import ensure_list

class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data apart from the filepath should be passed to the
    constructor of the ``DatasetReader``.
    """
    def instance_generator(self, file_path) -> InstanceGenerator:
        """
        The default implementation of this caches the instances in a list.
        If you want a lazy dataset that doesn't get loaded into memory all
        at once, you'll need to override this method.
        """
        iterable = self.read(file_path)

        # If `iterable` is already a list, this is a no-op.
        instances = ensure_list(iterable)

        # Each call to the returned `InstanceGenerator` just returns the list
        # of instances. If you modify that list (e.g. by shuffling),
        # those changes will persist to future calls.
        return lambda: instances

    def read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads in the instances from the given file_path and returns an `Iterable`
        (which could be a list or could be a generator).
        """
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DatasetReader':
        """
        Static method that constructs the dataset reader described by ``params``.
        """
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)

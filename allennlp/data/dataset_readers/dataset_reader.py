from typing import Iterator

from allennlp.data.dataset import Dataset, InstanceCollection, LazyDataset
from allennlp.data.instance import Instance
from allennlp.common import Params
from allennlp.common.registrable import Registrable


class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs an
    :class:`InstanceCollection`.  All parameters necessary to read the data apart from the filepath
    should be passed to the constructor of the ``DatasetReader``.
    """
    def __init__(self, lazy: bool = False) -> None:
        self._lazy = lazy

    def read(self, file_path: str) -> InstanceCollection:
        """
        Actually reads some data from the `file_path` and returns an :class:`InstanceCollection`.

        The base class implementation passes this off to :func:`_read_instances`, and then uses the
        resulting iterator to create either a :class:`~allennlp.data.dataset.Dataset` (which is
        `not` lazy, and stores all of the instances in memory) or a
        :class:`~allennlp.data.dataset.LazyDataset` (which `is` lazy, and reads, tokenizes, and
        indexes the instances at every pass through the data).

        If you want your ``DatasetReader`` to have this kind of lazy option built in, you should
        override :func:`_read_instances`.  Otherwise, you can just override this method, and that
        will still work fine with the API.
        """
        if self._lazy:
            return LazyDataset(lambda: self._read_instances(file_path))
        return Dataset(list(self._read_instances(file_path)))

    def _read_instances(self, file_path: str) -> Iterator[Instance]:
        """
        A helper method for easily adding lazy functionality to your ``DatasetReader``.  Instead of
        implementing :func:`read` directly and constructing a list of instances that goes into a
        :class:`Dataset`, you can implement this method that returns an ``Iterator`` over
        instances.  Where you would have added an instance to a list in :func:`read`, you can
        instead just ``yield`` the instance from this method, and then you get lazy behavior for
        free from the superclass by passing the ``lazy`` flag when calling the superclass
        constructor.

        The default implementation here raises a ``RuntimeError`` instead of a
        ``NotImplementedError``, because it is not required that a subclass implement this method.
        """
        # pylint: disable=no-self-use,unused-argument
        raise RuntimeError("Not implemented!")

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

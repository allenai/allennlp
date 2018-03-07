from typing import Iterable, Iterator, Callable
import logging

from allennlp.data.instance import Instance
from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """
    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) -> None:
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self) -> Iterator[Instance]:
        instances = self.instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")
        return instances

class DatasetReader(Registrable):
    """
    A ``DatasetReader`` knows how to turn a file containing a dataset into a collection
    of ``Instance`` s.  To implement your own, just override the `_read(file_path)` method
    to return an ``Iterable`` of the instances. This could be a list containing the instances
    or a lazy generator that returns them one at a time.

    All parameters necessary to _read the data apart from the filepath should be passed
    to the constructor of the ``DatasetReader``.

    Parameters
    ----------
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """
    def __init__(self, lazy: bool = False) -> None:
        self.lazy = lazy

    def read(self, file_path: str) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if lazy:
            return _LazyInstances(lambda: iter(self._read(file_path)))
        else:
            instances = self._read(file_path)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))
            return instances

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
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

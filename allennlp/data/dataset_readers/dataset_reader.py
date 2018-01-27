from typing import Iterable, Iterator

from allennlp.data.instance import Instance
from allennlp.common import Params
from allennlp.common.registrable import Registrable
from allennlp.common.util import ensure_list

class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs an :class:`InstanceGenerator`
    that returns an ``Iterable`` of the dataset's instances each time it's called. All parameters
    necessary to _read the data apart from the filepath should be passed to the constructor of the
    ``DatasetReader``.
    """
    lazy = False

    def instances(self, file_path) -> Iterable[Instance]:
        """
        If ``self.lazy`` is False, this caches all the instances
        from ``self._read()`` in a list and returns that list
        each time it's called.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``_read()`` each iteration.
        """
        if self.lazy:
            _read_thunk = lambda: self._read(file_path)

            class _LazyIterable(Iterable):
                def __iter__(self) -> Iterator[Instance]:
                    return iter(_read_thunk())

            return _LazyIterable()
        else:
            iterable = self._read(file_path)

            # If `iterable` is already a list, this is a no-op.
            instances = ensure_list(iterable)

            # Each call to the returned `InstanceGenerator` just returns
            # the list of instances. If you modify that list (e.g. by shuffling),
            # those changes will persist to future calls.
            return instances

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
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

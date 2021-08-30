"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""
import collections
import copy
import itertools
import json
import logging
import random
import re
import weakref
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Optional,
    Any,
    Set,
    List,
    Dict,
    Type,
    Union,
    cast,
    TypeVar,
    Generic,
    Iterable,
    Tuple,
    MutableMapping,
    Iterator,
    MutableSet,
    OrderedDict,
    Callable,
)

from allennlp.common.det_hash import det_hash

try:
    from typing import get_origin, get_args
except ImportError:

    def get_origin(tp):  # type: ignore
        return getattr(tp, "__origin__", None)

    def get_args(tp):  # type: ignore
        return getattr(tp, "__args__", ())


from allennlp.common import Registrable, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.from_params import (
    pop_and_construct_arg,
    infer_method_params,
    infer_constructor_params,
)
from allennlp.common.logging import AllenNlpLogger
from allennlp.tango.format import Format, DillFormat

logger = logging.getLogger(__name__)

_version_re = re.compile("""^[a-zA-Z0-9]+$""")

T = TypeVar("T")


class StepCache(Registrable):
    """This is a mapping from instances of `Step` to the results of that step."""

    def __contains__(self, step: object) -> bool:
        """This is a generic implementation of __contains__. If you are writing your own
        `StepCache`, you might want to write a faster one yourself."""
        if not isinstance(step, Step):
            return False
        try:
            self.__getitem__(step)
            return True
        except KeyError:
            return False

    @abstractmethod
    def __getitem__(self, step: "Step") -> Any:
        """Returns the results for the given step."""
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, step: "Step", value: Any) -> None:
        """Writes the results for the given step. Throws an exception if the step is already cached."""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of results saved in this cache."""
        raise NotImplementedError()

    def path_for_step(self, step: "Step") -> Optional[Path]:
        """Steps that can be restarted (like a training job that gets interrupted half-way through)
        must save their state somewhere. A `StepCache` can help by providing a suitable location
        in this method."""
        return None


@StepCache.register("memory")
class MemoryStepCache(StepCache):
    """This is a `StepCache` that stores results in memory. It is little more than a Python dictionary."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def __getitem__(self, step: "Step") -> Any:
        return self.cache[step.unique_id()]

    def __setitem__(self, step: "Step", value: Any) -> None:
        if step in self:
            raise ValueError(f"{step.unique_id()} is already cached! Will not overwrite.")
        if step.cache_results:
            self.cache[step.unique_id()] = value
        else:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)

    def __contains__(self, step: object):
        if isinstance(step, Step):
            return step.unique_id() in self.cache
        else:
            return False

    def __len__(self) -> int:
        return len(self.cache)


default_step_cache = MemoryStepCache()


@StepCache.register("directory")
class DirectoryStepCache(StepCache):
    """This is a `StepCache` that stores its results on disk, in the location given in `dir`.

    Every cached step gets a directory under `dir` with that step's `unique_id()`. In that
    directory we store the results themselves in some format according to the step's `FORMAT`,
    and we also write a `metadata.json` file that stores some metadata. The presence of
    `metadata.json` signifies that the cache entry is complete and has been written successfully.
    """

    LRU_CACHE_MAX_SIZE = 8

    def __init__(self, dir: Union[str, PathLike]):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        # We keep an in-memory cache as well so we don't have to de-serialize stuff
        # we happen to have in memory already.
        self.weak_cache: MutableMapping[str, Any] = weakref.WeakValueDictionary()

        # Not all Python objects can be referenced weakly, and even if they can they
        # might get removed too quickly, so we also keep an LRU cache.
        self.strong_cache: OrderedDict[str, Any] = collections.OrderedDict()

    def _add_to_cache(self, key: str, o: Any) -> None:
        if hasattr(o, "__next__"):
            # We never cache iterators, because they are mutable, storing their current position.
            return

        self.strong_cache[key] = o
        self.strong_cache.move_to_end(key)
        while len(self.strong_cache) > self.LRU_CACHE_MAX_SIZE:
            del self.strong_cache[next(iter(self.strong_cache))]

        try:
            self.weak_cache[key] = o
        except TypeError:
            pass  # Many native Python objects cannot be referenced weakly, and they throw TypeError when you try

    def _get_from_cache(self, key: str) -> Optional[Any]:
        result = self.strong_cache.get(key)
        if result is not None:
            self.strong_cache.move_to_end(key)
            return result
        try:
            return self.weak_cache[key]
        except KeyError:
            return None

    def __contains__(self, step: object) -> bool:
        if isinstance(step, Step):
            key = step.unique_id()
            if key in self.strong_cache:
                return True
            if key in self.weak_cache:
                return True
            metadata_file = self.path_for_step(step) / "metadata.json"
            return metadata_file.exists()
        else:
            return False

    def __getitem__(self, step: "Step") -> Any:
        key = step.unique_id()
        result = self._get_from_cache(key)
        if result is None:
            if step not in self:
                raise KeyError(step)
            result = step.format.read(self.path_for_step(step))
            self._add_to_cache(key, result)
        return result

    def __setitem__(self, step: "Step", value: Any) -> None:
        location = self.path_for_step(step)
        location.mkdir(parents=True, exist_ok=True)

        metadata_location = location / "metadata.json"
        if metadata_location.exists():
            raise ValueError(f"{metadata_location} already exists! Will not overwrite.")
        temp_metadata_location = metadata_location.with_suffix(".temp")

        try:
            step.format.write(value, location)
            metadata = {
                "step": step.unique_id(),
                "checksum": step.format.checksum(location),
            }
            with temp_metadata_location.open("wt") as f:
                json.dump(metadata, f)
            self._add_to_cache(step.unique_id(), value)
            temp_metadata_location.rename(metadata_location)
        except:  # noqa: E722
            temp_metadata_location.unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*/metadata.json"))

    def path_for_step(self, step: "Step") -> Path:
        return self.dir / step.unique_id()


class Step(Registrable, Generic[T]):
    """
    This class defines one step in your experiment. To write your own step, just derive from this class
    and overwrite the `run()` method. The `run()` method must have parameters with type hints.

    `Step.__init__()` takes all the arguments we want to run the step with. They get passed
    to `Step.run()` (almost) as they are. If the arguments are other instances of `Step`, those
    will be replaced with the step's results before calling `run()`. Further, there are four special
    parameters:

    * `step_name` contains an optional human-readable name for the step. This name is used for
      error messages and the like, and has no consequence on the actual computation.
    * `cache_results` specifies whether the results of this step should be cached. If this is
      `False`, the step is recomputed every time it is needed. If this is not set at all,
      we cache if the step is marked as `DETERMINISTIC`, and we don't cache otherwise.
    * `step_format` gives you a way to override the step's default format (which is given in `FORMAT`).
    * `only_if_needed` specifies whether we can skip this step if no other step depends on it. The
      default for this setting is to set it for all steps that don't have an explicit name.
    """

    default_implementation = "ref"

    DETERMINISTIC: bool = False
    """This describes whether this step can be relied upon to produce the same results every time
    when given the same inputs. If this is `False`, the step can't be cached, and neither can any
    step that depends on it."""

    CACHEABLE: Optional[bool] = None
    """This provides a direct way to turn off caching. For example, a step that reads a HuggingFace
    dataset doesn't need to be cached, because HuggingFace datasets already have their own caching
    mechanism. But it's still a deterministic step, and all following steps are allowed to cache.
    If it is `None`, the step figures out by itself whether it should be cacheable or not."""

    VERSION: Optional[str] = None
    """This is optional, but recommended. Specifying a version gives you a way to tell AllenNLP that
    a step has changed during development, and should now be recomputed. This doesn't invalidate
    the old results, so when you revert your code, the old cache entries will stick around and be
    picked up."""

    FORMAT: Format = DillFormat("gz")
    """This specifies the format the results of this step will be serialized in. See the documentation
    for `Format` for details."""

    def __init__(
        self,
        step_name: Optional[str] = None,
        cache_results: Optional[bool] = None,
        step_format: Optional[Format] = None,
        only_if_needed: Optional[bool] = None,
        **kwargs,
    ):
        self.logger = cast(AllenNlpLogger, logging.getLogger(self.__class__.__name__))

        if self.VERSION is not None:
            assert _version_re.match(
                self.VERSION
            ), f"Invalid characters in version '{self.VERSION}'"
        self.kwargs = kwargs

        if step_format is None:
            self.format = self.FORMAT
            if isinstance(self.format, type):
                self.format = self.format()
        else:
            self.format = step_format

        self.unique_id_cache: Optional[str] = None
        if step_name is None:
            self.name = self.unique_id()
        else:
            self.name = step_name

        if cache_results is True:
            if not self.CACHEABLE:
                raise ConfigurationError(
                    f"Step {self.name} is configured to use the cache, but it's not a cacheable step."
                )
            if not self.DETERMINISTIC:
                logger.warning(
                    f"Step {self.name} is going to be cached despite not being deterministic."
                )
            self.cache_results = True
        elif cache_results is False:
            self.cache_results = False
        elif cache_results is None:
            c = (self.DETERMINISTIC, self.CACHEABLE)
            if c == (False, None):
                self.cache_results = False
            elif c == (True, None):
                self.cache_results = True
            elif c == (False, False):
                self.cache_results = False
            elif c == (True, False):
                self.cache_results = False
            elif c == (False, True):
                logger.warning(
                    f"Step {self.name} is set to be cacheable despite not being deterministic."
                )
                self.cache_results = True
            elif c == (True, True):
                self.cache_results = True
            else:
                assert False, "Step.DETERMINISTIC or step.CACHEABLE are set to an invalid value."
        else:
            raise ConfigurationError(
                f"Step {self.name}'s cache_results parameter is set to an invalid value."
            )

        if step_name is None:
            self.only_if_needed = True
        else:
            self.only_if_needed = not self.cache_results
        if only_if_needed is not None:
            self.only_if_needed = only_if_needed

        self.work_dir_for_run: Optional[
            Path
        ] = None  # This is set only while the run() method runs.

    @classmethod
    def from_params(
        cls: Type["Step"],
        params: Params,
        constructor_to_call: Callable[..., "Step"] = None,
        constructor_to_inspect: Union[Callable[..., "Step"], Callable[["Step"], None]] = None,
        existing_steps: Optional[Dict[str, "Step"]] = None,
        step_name: Optional[str] = None,
        **extras,
    ) -> "Step":
        # Why do we need a custom from_params? Step classes have a run() method that takes all the
        # parameters necessary to perform the step. The __init__() method of the step takes those
        # same parameters, but each of them could be wrapped in another Step instead of being
        # supplied directly. from_params() doesn't know anything about these shenanigans, so
        # we have to supply the necessary logic here.

        if constructor_to_call is not None:
            raise ConfigurationError(
                f"{cls.__name__}.from_params cannot be called with a constructor_to_call."
            )
        if constructor_to_inspect is not None:
            raise ConfigurationError(
                f"{cls.__name__}.from_params cannot be called with a constructor_to_inspect."
            )

        if existing_steps is None:
            existing_steps = {}

        if isinstance(params, str):
            params = Params({"type": params})

        if not isinstance(params, Params):
            raise ConfigurationError(
                "from_params was passed a `params` object that was not a `Params`. This probably "
                "indicates malformed parameters in a configuration file, where something that "
                "should have been a dictionary was actually a list, or something else. "
                f"This happened when constructing an object of type {cls}."
            )

        as_registrable = cast(Type[Registrable], cls)
        choice = params.pop_choice(
            "type", choices=as_registrable.list_available(), default_to_first_choice=True
        )
        subclass, constructor_name = as_registrable.resolve_class_name(choice)
        if not issubclass(subclass, Step):
            # This can happen if `choice` is a fully qualified name.
            raise ConfigurationError(
                f"Tried to make a Step of type {choice}, but ended up with a {subclass}."
            )

        parameters = infer_method_params(subclass, subclass.run)
        del parameters["self"]
        init_parameters = infer_constructor_params(subclass)
        del init_parameters["self"]
        del init_parameters["kwargs"]
        parameter_overlap = parameters.keys() & init_parameters.keys()
        assert len(parameter_overlap) <= 0, (
            f"If this assert fails it means that you wrote a Step with a run() method that takes one of the "
            f"reserved parameters ({', '.join(init_parameters.keys())})"
        )
        parameters.update(init_parameters)

        kwargs: Dict[str, Any] = {}
        accepts_kwargs = False
        for param_name, param in parameters.items():
            if param.kind == param.VAR_KEYWORD:
                # When a class takes **kwargs we store the fact that the method allows extra keys; if
                # we get extra parameters, instead of crashing, we'll just pass them as-is to the
                # constructor, and hope that you know what you're doing.
                accepts_kwargs = True
                continue

            explicitly_set = param_name in params
            constructed_arg = pop_and_construct_arg(
                subclass.__name__,
                param_name,
                param.annotation,
                param.default,
                params,
                existing_steps=existing_steps,
                **extras,
            )

            # If the param wasn't explicitly set in `params` and we just ended up constructing
            # the default value for the parameter, we can just omit it.
            # Leaving it in can cause issues with **kwargs in some corner cases, where you might end up
            # with multiple values for a single parameter (e.g., the default value gives you lazy=False
            # for a dataset reader inside **kwargs, but a particular dataset reader actually hard-codes
            # lazy=True - the superclass sees both lazy=True and lazy=False in its constructor).
            if explicitly_set or constructed_arg is not param.default:
                kwargs[param_name] = constructed_arg

        if accepts_kwargs:
            kwargs.update(params)
        else:
            params.assert_empty(subclass.__name__)

        return subclass(step_name=step_name, **kwargs)

    @abstractmethod
    def run(self, **kwargs) -> T:
        """This is the main method of a step. Overwrite this method to define your step's action."""
        raise NotImplementedError()

    def _run_with_work_dir(self, cache: StepCache, **kwargs) -> T:
        if self.work_dir_for_run is not None:
            raise ValueError("You can only run a Step's run() method once at a time.")

        logger.info("Starting run for step %s of type %s", self.name, self.__class__.__name__)

        if self.DETERMINISTIC:
            random.seed(784507111)

            try:
                import numpy

                numpy.random.seed(784507111)
            except ImportError:
                pass

            try:
                import torch

                torch.manual_seed(784507111)
            except ImportError:
                pass

        step_dir = cache.path_for_step(self)
        if step_dir is None:
            work_dir = TemporaryDirectory(prefix=self.unique_id() + "-", suffix=".work")
            self.work_dir_for_run = Path(work_dir.name)
            try:
                return self.run(**kwargs)
            finally:
                self.work_dir_for_run = None
                work_dir.cleanup()
        else:
            self.work_dir_for_run = step_dir / "work"
            try:
                self.work_dir_for_run.mkdir(exist_ok=True, parents=True)
                return self.run(**kwargs)
            finally:
                # No cleanup, as we want to keep the directory for restarts or serialization.
                self.work_dir_for_run = None

    def work_dir(self) -> Path:
        """
        Returns a work directory that a step can use while its `run()` method runs.

        This directory stays around across restarts. You cannot assume that it is empty when your
        step runs, but you can use it to store information that helps you restart a step if it
        got killed half-way through the last time it ran."""
        if self.work_dir_for_run is None:
            raise ValueError("You can only call this method while the step is running.")
        return self.work_dir_for_run

    @classmethod
    def _replace_steps_with_results(cls, o: Any, cache: StepCache):
        if isinstance(o, Step):
            return o.result(cache)
        elif isinstance(o, list):
            return [cls._replace_steps_with_results(i, cache) for i in o]
        elif isinstance(o, tuple):
            return tuple(cls._replace_steps_with_results(list(o), cache))
        elif isinstance(o, set):
            return {cls._replace_steps_with_results(i, cache) for i in o}
        elif isinstance(o, dict):
            return {key: cls._replace_steps_with_results(value, cache) for key, value in o.items()}
        else:
            return o

    def result(self, cache: Optional[StepCache] = None) -> T:
        """Returns the result of this step. If the results are cached, it returns those. Otherwise it
        runs the step and returns the result from there."""
        if cache is None:
            cache = default_step_cache
        if self in cache:
            return cache[self]

        kwargs = self._replace_steps_with_results(self.kwargs, cache)
        result = self._run_with_work_dir(cache, **kwargs)
        if self.cache_results:
            cache[self] = result
            if hasattr(result, "__next__"):
                assert isinstance(result, Iterator)
                # Caching the iterator will consume it, so we write it to the cache and then read from the cache
                # for the return value.
                return cache[self]
        return result

    def ensure_result(self, cache: Optional[StepCache] = None) -> None:
        """This makes sure that the result of this step is in the cache. It does
        not return the result."""
        if not self.cache_results:
            raise ValueError(
                "It does not make sense to call ensure_result() on a step that's not cacheable."
            )

        if cache is None:
            cache = default_step_cache
        if self in cache:
            return

        kwargs = self._replace_steps_with_results(self.kwargs, cache)
        result = self._run_with_work_dir(cache, **kwargs)
        cache[self] = result

    def det_hash_object(self) -> Any:
        return self.unique_id()

    def unique_id(self) -> str:
        """Returns the unique ID for this step.

        Unique IDs are of the shape `$class_name-$version-$hash`, where the hash is the hash of the
        inputs for deterministic steps, and a random string of characters for non-deterministic ones."""
        if self.unique_id_cache is None:
            self.unique_id_cache = self.__class__.__name__
            if self.VERSION is not None:
                self.unique_id_cache += "-"
                self.unique_id_cache += self.VERSION

            self.unique_id_cache += "-"
            if self.DETERMINISTIC:
                self.unique_id_cache += det_hash(
                    (
                        (self.format.__class__.__module__, self.format.__class__.__qualname__),
                        self.format.VERSION,
                        self.kwargs,
                    )
                )[:32]
            else:
                self.unique_id_cache += det_hash(random.getrandbits((58 ** 32).bit_length()))[:32]

        return self.unique_id_cache

    def __hash__(self):
        return hash(self.unique_id())

    def __eq__(self, other):
        if isinstance(other, Step):
            return self.unique_id() == other.unique_id()
        else:
            return False

    def _ordered_dependencies(self) -> Iterable["Step"]:
        def dependencies_internal(o: Any) -> Iterable[Step]:
            if isinstance(o, Step):
                yield o
            elif isinstance(o, str):
                return  # Confusingly, str is an Iterable of itself, resulting in infinite recursion.
            elif isinstance(o, Iterable):
                yield from itertools.chain(*(dependencies_internal(i) for i in o))
            elif isinstance(o, dict):
                yield from dependencies_internal(o.values())
            else:
                return

        return dependencies_internal(self.kwargs.values())

    def dependencies(self) -> Set["Step"]:
        """Returns a set of steps that this step depends on.

        Does not return recursive dependencies."""
        return set(self._ordered_dependencies())

    def recursive_dependencies(self) -> Set["Step"]:
        """Returns a set of steps that this step depends on.

        This returns recursive dependencies."""

        seen = set()
        steps = list(self.dependencies())
        while len(steps) > 0:
            step = steps.pop()
            if step in seen:
                continue
            seen.add(step)
            steps.extend(step.dependencies())
        return seen


@Step.register("ref")
class _RefStep(Step[T], Generic[T]):
    def run(self, *, ref: str) -> T:  # type: ignore
        raise ConfigurationError(
            f"Step {self.name} is a RefStep (referring to {ref}). RefSteps cannot be executed. "
            "They are only useful while parsing an experiment."
        )

    def ref(self) -> str:
        return self.kwargs["ref"]

    def det_hash_object(self) -> Any:
        # If we're using a RefStep to compute a unique ID, something has gone wrong. The unique ID would
        # change once the RefStep is replaced with the actual step. Unique IDs are never supposed to
        # change.
        raise ValueError("Cannot compute hash of a _RefStep object.")

    class MissingStepError(Exception):
        def __init__(self, ref: str):
            self.ref = ref


def step_graph_from_params(params: Dict[str, Params]) -> Dict[str, Step]:
    """Given a mapping from strings to `Params` objects, this parses each `Params` object
    into a `Step`, and resolved dependencies between the steps. Returns a dictionary
    mapping step names to instances of `Step`."""

    # This algorithm for resolving step dependencies is O(n^2). Since we're
    # anticipating the number of steps to be in the dozens at most, we choose
    # simplicity over cleverness.
    unparsed_steps: Dict[str, Params] = params
    next_unparsed_steps: Dict[str, Params] = {}
    parsed_steps: Dict[str, Step] = {}
    steps_parsed = 0
    while len(unparsed_steps) > 0 or len(next_unparsed_steps) > 0:
        if len(unparsed_steps) <= 0:
            if steps_parsed <= 0:
                raise ConfigurationError(
                    f"Cannot parse steps {','.join(next_unparsed_steps.keys())}. Do you have a "
                    f"circle in your steps, or are you referring to a step that doesn't exist?"
                )
            unparsed_steps = next_unparsed_steps
            next_unparsed_steps = {}
            steps_parsed = 0
        step_name, step_params = unparsed_steps.popitem()
        if step_name in parsed_steps:
            raise ConfigurationError(f"Duplicate step name {step_name}")
        step_params_backup = copy.deepcopy(step_params)
        try:
            parsed_steps[step_name] = Step.from_params(
                step_params, existing_steps=parsed_steps, step_name=step_name
            )
            steps_parsed += 1
        except _RefStep.MissingStepError:
            next_unparsed_steps[step_name] = step_params_backup

    # Sanity-check the graph
    for step in parsed_steps.values():
        if step.cache_results:
            nondeterministic_dependencies = [
                s for s in step.recursive_dependencies() if not s.DETERMINISTIC
            ]
            if len(nondeterministic_dependencies) > 0:
                nd_step = nondeterministic_dependencies[0]
                logger.warning(
                    f"Task {step.name} is set to cache results, but depends on non-deterministic "
                    f"step {nd_step.name}. This will produce confusing results."
                )
                # We show this warning only once.
                break

    return parsed_steps


def tango_dry_run(
    step_or_steps: Union[Step, Iterable[Step]], step_cache: Optional[StepCache]
) -> List[Tuple[Step, bool]]:
    """
    Returns the list of steps that will be run, or read from cache, if you call
    a step's `result()` method.

    Steps come out as tuples `(step, read_from_cache)`, so you can see which
    steps will be read from cache, and which have to be run.
    """
    if isinstance(step_or_steps, Step):
        steps = [step_or_steps]
    else:
        steps = list(step_or_steps)

    cached_steps: MutableSet[Step]
    if step_cache is None:
        cached_steps = set()
    else:

        class SetWithFallback(set):
            def __contains__(self, item):
                return item in step_cache or super().__contains__(item)

        cached_steps = SetWithFallback()

    result = []
    seen_steps = set()
    steps.reverse()
    while len(steps) > 0:
        step = steps.pop()
        if step in seen_steps:
            continue
        dependencies = [s for s in step._ordered_dependencies() if s not in seen_steps]
        if len(dependencies) <= 0:
            result.append((step, step in cached_steps))
            cached_steps.add(step)
            seen_steps.add(step)
        else:
            steps.append(step)
            steps.extend(dependencies)

    return result

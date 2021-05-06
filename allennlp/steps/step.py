import copy
import inspect
import itertools
import logging
import random
import re
from typing import (
    Optional,
    Any,
    Set,
    List,
    Dict,
    Type,
    Callable,
    Union,
    cast,
    TypeVar,
    Generic,
    Iterable,
    MutableMapping,
)

from allennlp.common import Registrable, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.from_params import (
    infer_params,
    pop_and_construct_arg,
)
from allennlp.common.util import hash_object
from allennlp.steps.format import Format, DillFormat

logger = logging.getLogger(__name__)

_version_re = re.compile("""^[a-zA-Z0-9]+$""")

T = TypeVar("T")


class Step(Registrable, Generic[T]):
    DETERMINISTIC: bool = False
    VERSION: Optional[str] = None
    FORMAT: Format = DillFormat("gz")

    def __init__(
        self,
        step_name: Optional[str] = None,
        cache_results: Optional[bool] = None,
        step_format: Optional[Format] = None,
        produce_results: bool = False,
        **kwargs,
    ):
        """
        `Step.__init__()` takes all the arguments we want to run the step with. They get passed
        to `Step.run()` (almost) as they are. If the arguments are other instances of `Step`, those
        will be replaced with the step's results before calling `run()`. Further, there are two special
        parameters:
        * `step_name` contains an optional human-readable name for the step. This name is used for
          error messages and the like, and has no consequence on the actual computation.
        * `cache_results` specifies whether the results of this step should be cached. If this is
          `False`, the step is recomputed every time it is needed. If this is not set at all,
          we cache if the step is marked as `DETERMINISTIC`, and we don't cache otherwise.
        """
        if self.VERSION is not None:
            assert _version_re.match(
                self.VERSION
            ), f"Invalid characters in version '{self.VERSION}'"
        self.name = step_name
        self.kwargs = kwargs

        self.unique_id_cache: Optional[str] = None
        if self.name is None:
            self.name = self.unique_id()

        self.produce_results = produce_results

        self.format = step_format
        if self.format is None:
            self.format = self.FORMAT

        if cache_results is None:
            cache_results = self.DETERMINISTIC
        self.cache_results = bool(cache_results)
        if self.cache_results and not self.DETERMINISTIC:
            logger.warning(
                f"Task {self.name} is going to be cached despite not being deterministic."
            )

    @classmethod
    def from_params(
        cls: Type["Step"],
        params: Params,
        constructor_to_call: Callable[..., "Step"] = None,
        constructor_to_inspect: Union[Callable[..., "Step"], Callable[["Step"], None]] = None,
        existing_steps: Optional[Dict[str, "Step"]] = None,
        **extras,
    ) -> "Step":
        # Why do we need a custom from_params? Step classes have a run() method that takes all the
        # parameters necessary to perform the step. The __init__() method of the step takes those
        # same parameters, but each of them could be wrapped in another Step instead of being
        # supplied directly. from_params() doesn't know anything about these shenanigans, so
        # we have to supply the necessary logic here.

        # TODO: Maybe we figure out later if we need this and what to do about it?
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
        choice = params.pop_choice("type", choices=as_registrable.list_available())
        subclass, constructor_name = as_registrable.resolve_class_name(choice)
        kwargs: Dict[str, Any] = {}
        parameters = infer_params(subclass, subclass.run)
        accepts_kwargs = False
        for param_name, param in parameters.items():
            # Skip "self". You're not *required* to call the first parameter "self",
            # so in theory this logic is fragile, but if you don't call the self parameter
            # "self" you kind of deserve what happens.
            if param_name == "self":
                continue

            if param.kind == param.VAR_KEYWORD:
                # When a class takes **kwargs, we do two things: first, we assume that the **kwargs are
                # getting passed to the super class, so we inspect super class constructors to get
                # allowed arguments (that happens in `infer_params` above).  Second, we store the fact
                # that the method allows extra keys; if we get extra parameters, instead of crashing,
                # we'll just pass them as-is to the constructor, and hope that you know what you're
                # doing.
                accepts_kwargs = True
                continue

            annotation = Union[Step[param.annotation], param.annotation]

            explicitly_set = param_name in params
            constructed_arg = pop_and_construct_arg(
                subclass.__name__, param_name, annotation, param.default, params, **extras
            )

            if isinstance(constructed_arg, str) and not issubclass(  # we found a string
                param.annotation, str
            ):  # we didn't want a string
                if constructed_arg in existing_steps:  # the string matches an existing task
                    constructed_arg = existing_steps[constructed_arg]
                else:
                    raise _RefStep.MissingStepError(constructed_arg)

            if isinstance(constructed_arg, Step):
                if isinstance(constructed_arg, _RefStep):
                    try:
                        constructed_arg = existing_steps[constructed_arg.ref()]
                    except KeyError:
                        raise _RefStep.MissingStepError(constructed_arg.ref())

                return_type = inspect.signature(constructed_arg.run).return_annotation
                if return_type == inspect.Signature.empty:
                    logger.warning(
                        "Step %s has no return type annotation. Those are really helpful when "
                        "debugging, so we recommend them highly.",
                        subclass.__name__,
                    )
                elif not issubclass(return_type, param.annotation):
                    raise ConfigurationError(
                        f"Step {constructed_arg.name} returns {return_type}, but "
                        f"{subclass.__name__} expects {param.annotation}."
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

        return subclass(**kwargs)

    def run(self, **kwargs) -> T:
        raise NotImplementedError

    def result(self, cache: Optional[MutableMapping["Step", Any]] = None):
        if cache is None:
            from allennlp.steps.step_cache import default_step_cache

            cache = default_step_cache
        if self in cache:
            return cache[self]

        def replace_steps_with_results(o: Any):
            if isinstance(o, Step):
                return o.result()
            elif isinstance(o, List):
                return [replace_steps_with_results(i) for i in o]
            elif isinstance(o, Set):
                return {replace_steps_with_results(i) for i in o}
            elif isinstance(o, Dict):
                return {key: replace_steps_with_results(value) for key, value in o.items()}
            else:
                return o

        kwargs = replace_steps_with_results(self.kwargs)
        result = self.run(**kwargs)
        if self.cache_results:
            # If we have an iterator as a result, we have to copy it into a list first,
            # otherwise we can't cache it.
            if hasattr(result, "__next__"):
                result = list(result)
            cache[self] = result
        return result

    def unique_id(self) -> str:
        if self.unique_id_cache is None:
            self.unique_id_cache = self.__class__.__name__
            if self.VERSION is not None:
                self.unique_id_cache += "-"
                self.unique_id_cache += self.VERSION

            self.unique_id_cache += "-"
            if self.DETERMINISTIC:

                def replace_steps_with_hashes(o: Any):
                    if isinstance(o, Step):
                        return o.unique_id()
                    elif isinstance(o, List):
                        return [replace_steps_with_hashes(i) for i in o]
                    elif isinstance(o, Set):
                        return {replace_steps_with_hashes(i) for i in o}
                    elif isinstance(o, Dict):
                        return {key: replace_steps_with_hashes(value) for key, value in o.items()}
                    else:
                        return o

                self.unique_id_cache += hash_object(replace_steps_with_hashes(self.kwargs))[:32]
            else:
                self.unique_id_cache += hash_object(random.getrandbits((58 ** 32).bit_length()))[
                    :32
                ]

        return self.unique_id_cache

    def __hash__(self):
        return hash(self.unique_id())

    def __eq__(self, other):
        if isinstance(self, Step):
            return self.unique_id() == other.unique_id()
        else:
            return False

    def dependencies(self) -> Set["Step"]:
        def dependencies_internal(o: Any) -> Iterable[Step]:
            if isinstance(o, Step):
                yield o
            elif isinstance(o, str):
                return  # Confusingly, str is an Iterable of itself, resulting in infinite recursion.
            elif isinstance(o, Iterable):
                yield from itertools.chain(*(dependencies_internal(i) for i in o))
            elif isinstance(o, Dict):
                yield from dependencies_internal(o.values())
            else:
                return

        return set(dependencies_internal(self.kwargs.values()))

    def recursive_dependencies(self) -> Set["Step"]:
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
class _RefStep(Step[T]):
    def run(self, ref: str) -> T:
        raise ConfigurationError(
            f"Step {self.name} is still a RefStep (referring to {ref}). RefSteps cannot be executed. "
            "They are only useful while parsing an experiment."
        )

    def ref(self) -> str:
        return self.kwargs["ref"]

    class MissingStepError(Exception):
        def __init__(self, ref: str):
            self.ref = ref


def step_graph_from_params(params: Dict[str, Params]) -> Dict[str, Step]:
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
                step_params, existing_steps=parsed_steps, extras={"step_name": step_name}
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

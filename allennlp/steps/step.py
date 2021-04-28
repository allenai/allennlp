import copy
import dataclasses
import inspect
import logging
import random
import re
from dataclasses import dataclass, field
from typing import (
    Optional,
    Mapping,
    Any,
    Set,
    Sequence,
    List,
    Dict,
    MutableMapping,
    Type,
    Callable,
    Union,
    cast,
    TypeVar,
    Generic,
)

import datasets

from allennlp.common import Registrable, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.from_params import (
    infer_params,
    pop_and_construct_arg,
)
from allennlp.common.util import hash_object
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)

_version_re = re.compile("""^[a-zA-Z0-9]+$""")


class StepCache(MutableMapping["Step", Any], Registrable):
    def __delitem__(self, key: "Step"):
        raise NotImplementedError("Cached results are forever.")

    def __iter__(self):
        raise NotImplementedError("Step caches are not iterable.")

    def __contains__(self, item: "Step"):
        raise NotImplementedError


@StepCache.register("memory")
class MemoryStepCache(StepCache):
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def __getitem__(self, step: "Step") -> Any:
        return self.cache.get(step.unique_id())

    def __setitem__(self, step: "Step", value: Any) -> None:
        if step.cache_results:
            self.cache[step.unique_id()] = value
        else:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)

    def __contains__(self, step: "Step"):
        return step.unique_id() in self.cache

    def __len__(self) -> int:
        return len(self.cache)


_default_step_cache = MemoryStepCache()


T = TypeVar("T")


class Step(Registrable, Generic[T]):
    DETERMINISTIC: bool = False
    VERSION: Optional[str] = None

    def __init__(self, **kwargs):
        if self.VERSION is not None:
            assert _version_re.match(
                self.VERSION
            ), f"Invalid characters in version '{self.VERSION}'"
        self.name = kwargs.pop("step_name", None)

        self.kwargs = kwargs
        self.unique_id_cache: Optional[str] = None
        if self.name is None:
            self.name = self.unique_id()

        self.cache_results = kwargs.pop("cache_results", self.DETERMINISTIC)
        self.cache_results = bool(self.cache_results)
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
                    raise RefStep.MissingStepError(constructed_arg)

            if isinstance(constructed_arg, Step):
                if isinstance(constructed_arg, RefStep):
                    try:
                        constructed_arg = existing_steps[constructed_arg.ref()]
                    except KeyError:
                        raise RefStep.MissingStepError(constructed_arg.ref())

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

    def result(self, cache: Optional[StepCache] = None):
        if cache is None:
            global _default_step_cache
            cache = _default_step_cache
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


Split = Sequence[Any]


@Step.register("ref")
class RefStep(Step[T]):
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
        except RefStep.MissingStepError:
            next_unparsed_steps[step_name] = step_params_backup

    return parsed_steps


@dataclass
class AllenNlpDataset:
    splits: Mapping[str, Split]
    vocab: Optional[Vocabulary] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@Step.register("huggingface_dataset")
class HuggingfaceDataset(Step):
    DETERMINISTIC = True
    VERSION = "001"

    def run(self, dataset_name: str) -> AllenNlpDataset:
        return AllenNlpDataset(datasets.load_dataset(dataset_name), None, {"source": "huggingface"})


@Step.register("text_only")
class TextOnlyDataset(Step):
    DETERMINISTIC = True

    def run(self, input: AllenNlpDataset, fields_to_keep: Set[str]) -> AllenNlpDataset:
        return dataclasses.replace(
            input,
            splits={
                split_name: [
                    {"text": field_value}
                    for instance in split
                    for field_name, field_value in instance.items()
                    if field_name in fields_to_keep
                ]
                for split_name, split in input.splits.items()
            },
        )

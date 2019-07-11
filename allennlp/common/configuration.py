"""
Tools for programmatically generating config files for AllenNLP models.
"""
# pylint: disable=protected-access,too-many-return-statements

from typing import NamedTuple, Optional, Any, List, TypeVar, Generic, Dict, Union, Sequence, Tuple
import collections
import json
import re

import torch
from numpydoc.docscrape import NumpyDocString

from allennlp.common import Registrable, JsonDict
from allennlp.nn.activations import Activation
from allennlp.nn.initializers import Initializer, PretrainedModelInitializer
from allennlp.nn.regularizers import Regularizer

def _remove_prefix(class_name: str) -> str:
    rgx = r"^(typing\.|builtins\.)"
    return re.sub(rgx, "", class_name)

def full_name(cla55: Optional[type]) -> str:
    """
    Return the full name (including module) of the given class.
    """
    # Special case to handle None:
    if cla55 is None:
        return "?"

    if issubclass(cla55, Initializer) and cla55 not in [Initializer, PretrainedModelInitializer]:
        init_fn = cla55()._init_function
        return f"{init_fn.__module__}.{init_fn.__name__}"

    origin = getattr(cla55, '__origin__', None)
    args = getattr(cla55, '__args__', ())

    # Special handling for compound types
    if origin in (Dict, dict):
        key_type, value_type = args
        return f"""Dict[{full_name(key_type)}, {full_name(value_type)}]"""
    elif origin in (Tuple, tuple, List, list, Sequence, collections.abc.Sequence):
        return f"""{_remove_prefix(str(origin))}[{", ".join(full_name(arg) for arg in args)}]"""
    elif origin == Union:
        # Special special case to handle optional types:
        if len(args) == 2 and args[-1] == type(None):
            return f"""Optional[{full_name(args[0])}]"""
        else:
            return f"""Union[{", ".join(full_name(arg) for arg in args)}]"""
    else:
        return _remove_prefix(f"{cla55.__module__}.{cla55.__name__}")


def json_annotation(cla55: Optional[type]):
    # Special case to handle None:
    if cla55 is None:
        return {'origin': '?'}

    # Special case to handle activation functions, which can't be specified as JSON
    if cla55 == Activation:
        return {'origin': 'str'}

    # Hack because e.g. typing.Union isn't a type.
    if isinstance(cla55, type) and issubclass(cla55, Initializer) and cla55 != Initializer:
        init_fn = cla55()._init_function
        return {'origin': f"{init_fn.__module__}.{init_fn.__name__}"}

    origin = getattr(cla55, '__origin__', None)
    args = getattr(cla55, '__args__', ())

    # Special handling for compound types
    if origin in (Dict, dict):
        key_type, value_type = args
        return {'origin': "Dict", 'args': [json_annotation(key_type), json_annotation(value_type)]}
    elif origin in (Tuple, tuple, List, list, Sequence, collections.abc.Sequence):
        return {'origin': _remove_prefix(str(origin)), 'args': [json_annotation(arg) for arg in args]}
    elif origin == Union:
        # Special special case to handle optional types:
        if len(args) == 2 and args[-1] == type(None):
            return json_annotation(args[0])
        else:
            return {'origin': "Union", 'args': [json_annotation(arg) for arg in args]}
    elif cla55 == Ellipsis:
        return {'origin': "..."}
    else:
        return {'origin': _remove_prefix(f"{cla55.__module__}.{cla55.__name__}")}


class ConfigItem(NamedTuple):
    """
    Each ``ConfigItem`` represents a single entry in a configuration JsonDict.
    """
    name: str
    annotation: type
    default_value: Optional[Any] = None
    comment: str = ''

    def to_json(self) -> JsonDict:
        json_dict = {
                "name": self.name,
                "annotation": json_annotation(self.annotation),
        }

        if is_configurable(self.annotation):
            json_dict["configurable"] = True

        if is_registrable(self.annotation):
            json_dict["registrable"] = True

        if self.default_value != NO_DEFAULT:
            try:
                # Ugly check that default value is actually serializable
                json.dumps(self.default_value)
                json_dict["defaultValue"] = self.default_value
            except TypeError:
                print(f"unable to json serialize {self.default_value}, using None instead")
                json_dict["defaultValue"] = None


        if self.comment:
            json_dict["comment"] = self.comment

        return json_dict


T = TypeVar("T")


class Config(Generic[T]):
    """
    A ``Config`` represents an entire subdict in a configuration file.
    If it corresponds to a named subclass of a registrable class,
    it will also contain a ``type`` item in addition to whatever
    items are required by the subclass ``from_params`` method.
    """
    def __init__(self, items: List[ConfigItem], typ3: str = None) -> None:
        self.items = items
        self.typ3 = typ3

    def __repr__(self) -> str:
        return f"Config({self.items})"

    def to_json(self) -> JsonDict:
        blob: JsonDict = {'items': [item.to_json() for item in self.items]}

        if self.typ3:
            blob["type"] = self.typ3

        return blob


# ``None`` is sometimes the default value for a function parameter,
# so we use a special sentinel to indicate that a parameter has no
# default value.
NO_DEFAULT = object()

def _get_config_type(cla55: type) -> Optional[str]:
    """
    Find the name (if any) that a subclass was registered under.
    We do this simply by iterating through the registry until we
    find it.
    """
    # Special handling for pytorch RNN types:
    if cla55 == torch.nn.RNN:
        return "rnn"
    elif cla55 == torch.nn.LSTM:
        return "lstm"
    elif cla55 == torch.nn.GRU:
        return "gru"

    for subclass_dict in Registrable._registry.values():
        for name, subclass in subclass_dict.items():
            if subclass == cla55:
                return name

        # Special handling for initializer functions
            if hasattr(subclass, '_initializer_wrapper'):
                sif = subclass()._init_function
                if sif == cla55:
                    return sif.__name__.rstrip("_")

    return None

def _docspec_comments(obj) -> Dict[str, str]:
    """
    Inspect the docstring and get the comments for each parameter.
    """
    # Sometimes our docstring is on the class, and sometimes it's on the initializer,
    # so we've got to check both.
    class_docstring = getattr(obj, '__doc__', None)
    init_docstring = getattr(obj.__init__, '__doc__', None) if hasattr(obj, '__init__') else None

    docstring = class_docstring or init_docstring or ''

    doc = NumpyDocString(docstring)
    params = doc["Parameters"]
    comments: Dict[str, str] = {}

    for line in params:
        # It looks like when there's not a space after the parameter name,
        # numpydocstring parses it incorrectly.
        name_bad = line[0]
        name = name_bad.split(":")[0]

        # Sometimes the line has 3 fields, sometimes it has 4 fields.
        comment = "\n".join(line[-1])

        comments[name] = comment

    return comments


def render_config(config: Config, indent: str = "") -> str:
    """
    Pretty-print a config in sort-of-JSON+comments.
    """
    # Add four spaces to the indent.
    new_indent = indent + "    "

    return "".join([
            # opening brace + newline
            "{\n",
            # "type": "...", (if present)
            f'{new_indent}"type": "{config.typ3}",\n' if config.typ3 else '',
            # render each item
            "".join(_render(item, new_indent) for item in config.items),
            # indent and close the brace
            indent,
            "}\n"
    ])


def _remove_optional(typ3: type) -> type:
    origin = getattr(typ3, '__origin__', None)
    args = getattr(typ3, '__args__', None)

    if origin == Union and len(args) == 2 and args[-1] == type(None):
        return _remove_optional(args[0])
    else:
        return typ3

def is_registrable(typ3: type) -> bool:
    # Throw out optional:
    typ3 = _remove_optional(typ3)

    # Anything with a from_params method is itself configurable.
    # So are regularizers even though they don't.
    if typ3 == Regularizer:
        return True

    # Some annotations are unions and will crash `issubclass`.
    # TODO: figure out a better way to deal with them
    try:
        return issubclass(typ3, Registrable)
    except TypeError:
        return False


def is_configurable(typ3: type) -> bool:
    # Throw out optional:
    typ3 = _remove_optional(typ3)

    # Anything with a from_params method is itself configurable.
    # So are regularizers even though they don't.
    return any([
            hasattr(typ3, 'from_params'),
            typ3 == Regularizer,
    ])

def _render(item: ConfigItem, indent: str = "") -> str:
    """
    Render a single config item, with the provided indent
    """
    optional = item.default_value != NO_DEFAULT

    if is_configurable(item.annotation):
        rendered_annotation = f"{item.annotation} (configurable)"
    else:
        rendered_annotation = str(item.annotation)

    rendered_item = "".join([
            # rendered_comment,
            indent,
            "// " if optional else "",
            f'"{item.name}": ',
            rendered_annotation,
            f" (default: {item.default_value} )" if optional else "",
            f" // {item.comment}" if item.comment else "",
            "\n"
    ])

    return rendered_item


ConfigTuple = Tuple[str, type, Any, str]  # pylint: disable=invalid-name

def configuration(config_items: List[Union[ConfigItem, ConfigTuple]]):
    """
    Decorator to associate a ``Config`` with a from_params method
    when it can't be inferred from the constructor signature.
    """
    def assign_config(method):
        setattr(method, '_config_items', config_items)
        return method

    return assign_config

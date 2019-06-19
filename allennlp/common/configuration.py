"""
Tools for programmatically generating config files for AllenNLP models.
"""
# pylint: disable=protected-access,too-many-return-statements

from typing import NamedTuple, Optional, Any, List, TypeVar, Generic, Type, Dict, Union, Sequence, Tuple
import collections
import inspect
import importlib
import json
import re

import torch
from numpydoc.docscrape import NumpyDocString

from allennlp.common import Registrable, JsonDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary, DEFAULT_NON_PADDED_NAMESPACES
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.activations import Activation
from allennlp.nn.initializers import Initializer, PretrainedModelInitializer
from allennlp.nn.regularizers import Regularizer
from allennlp.training.optimizers import Optimizer as AllenNLPOptimizer
from allennlp.training.trainer import Trainer

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

        if self.default_value != _NO_DEFAULT:
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
_NO_DEFAULT = object()

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

def _auto_config(cla55: Type[T]) -> Config[T]:
    """
    Create the ``Config`` for a class by reflecting on its ``__init__``
    method and applying a few hacks.
    """
    typ3 = _get_config_type(cla55)

    # Don't include self, or vocab
    names_to_ignore = {"self", "vocab"}

    # Hack for RNNs
    if cla55 in [torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU]:
        cla55 = torch.nn.RNNBase
        names_to_ignore.add("mode")

    if isinstance(cla55, type):
        # It's a class, so inspect its constructor
        function_to_inspect = cla55.__init__
    else:
        # It's a function, so inspect it, and ignore tensor
        function_to_inspect = cla55
        names_to_ignore.add("tensor")

    argspec = inspect.getfullargspec(function_to_inspect)
    comments = _docspec_comments(cla55)

    items: List[ConfigItem] = []

    num_args = len(argspec.args)
    defaults = list(argspec.defaults or [])
    num_default_args = len(defaults)
    num_non_default_args = num_args - num_default_args

    # Required args all come first, default args at the end.
    defaults = [_NO_DEFAULT for _ in range(num_non_default_args)] + defaults

    for name, default in zip(argspec.args, defaults):
        if name in names_to_ignore:
            continue
        annotation = argspec.annotations.get(name)
        comment = comments.get(name)

        # Don't include Model, the only place you'd specify that is top-level.
        if annotation == Model:
            continue

        # Don't include DataIterator, the only place you'd specify that is top-level.
        if annotation == DataIterator:
            continue

        # Don't include params for an Optimizer
        if torch.optim.Optimizer in getattr(cla55, '__bases__', ()) and name == "params":
            continue

        # Don't include datasets in the trainer
        if cla55 == Trainer and name.endswith("_dataset"):
            continue

        # Hack in our Optimizer class to the trainer
        if cla55 == Trainer and annotation == torch.optim.Optimizer:
            annotation = AllenNLPOptimizer

        # Hack in embedding num_embeddings as optional (it can be inferred from the pretrained file)
        if cla55 == Embedding and name == "num_embeddings":
            default = None

        items.append(ConfigItem(name, annotation, default, comment))

    # More hacks, Embedding
    if cla55 == Embedding:
        items.insert(1, ConfigItem("pretrained_file", str, None))

    return Config(items, typ3=typ3)


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
    optional = item.default_value != _NO_DEFAULT

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

BASE_CONFIG: Config = Config([
        ConfigItem(name="dataset_reader",
                   annotation=DatasetReader,
                   default_value=_NO_DEFAULT,
                   comment="specify your dataset reader here"),
        ConfigItem(name="validation_dataset_reader",
                   annotation=DatasetReader,
                   default_value=None,
                   comment="same as dataset_reader by default"),
        ConfigItem(name="train_data_path",
                   annotation=str,
                   default_value=_NO_DEFAULT,
                   comment="path to the training data"),
        ConfigItem(name="validation_data_path",
                   annotation=str,
                   default_value=None,
                   comment="path to the validation data"),
        ConfigItem(name="test_data_path",
                   annotation=str,
                   default_value=None,
                   comment="path to the test data (you probably don't want to use this!)"),
        ConfigItem(name="evaluate_on_test",
                   annotation=bool,
                   default_value=False,
                   comment="whether to evaluate on the test dataset at the end of training (don't do it!)"),
        ConfigItem(name="model",
                   annotation=Model,
                   default_value=_NO_DEFAULT,
                   comment="specify your model here"),
        ConfigItem(name="iterator",
                   annotation=DataIterator,
                   default_value=_NO_DEFAULT,
                   comment="specify your data iterator here"),
        ConfigItem(name="trainer",
                   annotation=Trainer,
                   default_value=_NO_DEFAULT,
                   comment="specify the trainer parameters here"),
        ConfigItem(name="datasets_for_vocab_creation",
                   annotation=List[str],
                   default_value=None,
                   comment="if not specified, use all datasets"),
        ConfigItem(name="vocabulary",
                   annotation=Vocabulary,
                   default_value=None,
                   comment="vocabulary options"),

])

def _valid_choices(cla55: type) -> Dict[str, str]:
    """
    Return a mapping {registered_name -> subclass_name}
    for the registered subclasses of `cla55`.
    """
    valid_choices: Dict[str, str] = {}

    if cla55 not in Registrable._registry:
        raise ValueError(f"{cla55} is not a known Registrable class")

    for name, subclass in Registrable._registry[cla55].items():
        # These wrapper classes need special treatment
        if isinstance(subclass, (_Seq2SeqWrapper, _Seq2VecWrapper)):
            subclass = subclass._module_class

        valid_choices[name] = full_name(subclass)

    return valid_choices

def choices(full_path: str = '') -> List[str]:
    parts = full_path.split(".")
    class_name = parts[-1]
    module_name = ".".join(parts[:-1])
    module = importlib.import_module(module_name)
    cla55 = getattr(module, class_name)
    return list(_valid_choices(cla55).values())


def configure(full_path: str = '') -> Config:
    if not full_path:
        return BASE_CONFIG

    parts = full_path.split(".")
    class_name = parts[-1]
    module_name = ".".join(parts[:-1])
    module = importlib.import_module(module_name)
    cla55 = getattr(module, class_name)
    if cla55 == Vocabulary:
        return VOCAB_CONFIG
    else:
        return _auto_config(cla55)


# ONE OFF LOGIC FOR VOCABULARY
VOCAB_CONFIG: Config = Config([
        ConfigItem(name="directory_path",
                   annotation=str,
                   default_value=None,
                   comment="path to an existing vocabulary (if you want to use one)"),
        ConfigItem(name="extend",
                   annotation=bool,
                   default_value=False,
                   comment="whether to extend the existing vocabulary (if you specified one)"),
        ConfigItem(name="min_count",
                   annotation=Dict[str, int],
                   default_value=None,
                   comment="only include tokens that occur at least this many times"),
        ConfigItem(name="max_vocab_size",
                   annotation=Union[int, Dict[str, int]],
                   default_value=None,
                   comment="used to cap the number of tokens in your vocabulary"),
        ConfigItem(name="non_padded_namespaces",
                   annotation=List[str],
                   default_value=DEFAULT_NON_PADDED_NAMESPACES,
                   comment="namespaces that don't get padding or OOV tokens"),
        ConfigItem(name="pretrained_files",
                   annotation=Dict[str, str],
                   default_value=None,
                   comment="pretrained embedding files for each namespace"),
        ConfigItem(name="min_pretrained_embeddings",
                   annotation=Dict[str, int],
                   default_value=None,
                   comment="specifies a number of lines to keep for each namespace, "
                   "even for words not appearing in the data"),
        ConfigItem(name="only_include_pretrained_words",
                   annotation=bool,
                   default_value=False,
                   comment=("if True, keeps only the words that appear in the pretrained set. "
                            "if False, also includes non-pretrained words that exceed min_count.")),
        ConfigItem(name="tokens_to_add",
                   annotation=Dict[str, List[str]],
                   default_value=None,
                   comment=("any tokens here will certainly be included in the keyed namespace, "
                            "regardless of your data"))
])

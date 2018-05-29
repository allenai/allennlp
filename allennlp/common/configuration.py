"""
Tools for programmatically generating config files for AllenNLP models.
"""
# pylint: disable=protected-access

from typing import NamedTuple, Optional, Any, List, TypeVar, Generic, Type, Dict
import inspect
import importlib

import torch

from allennlp.common import Registrable
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper
from allennlp.training.optimizers import Optimizer as AllenNLPOptimizer
from allennlp.training.trainer import Trainer

JsonDict = Dict[str, Any]  # pylint: disable=invalid-name

def full_name(cla55: type) -> str:
    return f"{cla55.__module__}.{cla55.__name__}"

class ConfigItem(NamedTuple):
    name: str
    annotation: type
    default_value: Optional[Any] = None
    comment: str = ''

    def to_json(self) -> JsonDict:
        return {
                "annotation": full_name(self.annotation),
                "default_value": str(self.default_value),
                "comment": self.comment
        }


T = TypeVar("T")


class Config(Generic[T]):
    def __init__(self, items: List[ConfigItem], typ3: str = None) -> None:
        self.items = items
        self.typ3 = typ3

    def __repr__(self) -> str:
        return f"Config({self.items})"

    def to_json(self) -> JsonDict:
        item_dict = {
                item.name: item.to_json
                for item in self.items
        }

        if self.typ3:
            item_dict["type"] = self.typ3

        return item_dict


NO_DEFAULT = object()

def get_config_type(cla55: type) -> Optional[str]:
    for subclass_dict in Registrable._registry.values():
        for name, subclass in subclass_dict.items():
            if subclass == cla55:
                return name
    return None


def auto_config(cla55: Type[T]) -> Config[T]:
    argspec = inspect.getfullargspec(cla55.__init__)

    items: List[ConfigItem] = []

    num_args = len(argspec.args)
    defaults = list(argspec.defaults or [])
    num_default_args = len(defaults)
    num_non_default_args = num_args - num_default_args
    defaults = [NO_DEFAULT for _ in range(num_non_default_args)] + defaults

    for name, default in zip(argspec.args, defaults):
        # Don't include self
        if name == "self":
            continue
        annotation = argspec.annotations.get(name)

        # Don't include Model, the only place you'd specify that is top-level.
        if annotation == Model:
            continue

        # Don't include params for an Optimizer
        if torch.optim.Optimizer in cla55.__bases__ and name == "params":
            continue

        # Don't include datasets in the trainer
        if cla55 == Trainer and name.endswith("_dataset"):
            continue

        # Hack in our Optimizer class to the trainer
        if cla55 == Trainer and annotation == torch.optim.Optimizer:
            annotation = AllenNLPOptimizer

        items.append(ConfigItem(name, annotation, default))

    return Config(items, typ3=get_config_type(cla55))

def render_config(config: Config, indent: str = ""):
    new_indent = indent + "   "

    return "".join([
            "{\n",
            f'{new_indent}"type": "{config.typ3}",\n' if config.typ3 else '',
            "".join(render(item, new_indent) for item in config.items),
            indent,
            "}\n"
    ])

def render(item: ConfigItem, indent: str = "") -> str:
    optional = item.default_value != NO_DEFAULT

    # if isinstance(item.annotation, Config):
    #     if optional:
    #         new_indent = indent + "//  "
    #     else:
    #         new_indent = indent + "    "
    #     rendered_annotation = render_config(item.annotation, new_indent)
    # else:
    #     rendered_annotation = str(item.annotation)

    if hasattr(item.annotation, 'from_params'):
        rendered_annotation = f"{item.annotation} (configurable)"
    else:
        rendered_annotation = str(item.annotation)

    rendered_comment = "".join([
            indent,
            "// ",
            item.comment,
            "\n"
    ]) if item.comment else ""

    rendered_item = "".join([
            rendered_comment,
            indent,
            "// " if optional else "",
            f'"{item.name}": ',
            rendered_annotation,
            f" (default: {item.default_value} )" if optional else "",
            f"// {item.comment}" if item.comment else "",
            "\n"
    ])

    return rendered_item

KNOWN_CONFIGS: Dict[str, Config] = {}

def valid_choices(cla55: type) -> Dict[str, str]:
    choices: Dict[str, str] = {}

    for name, subclass in Registrable._registry[cla55].items():
        # These wrapper classes need special treatment
        if isinstance(subclass, (_Seq2SeqWrapper, _Seq2VecWrapper)):
            subclass = subclass._module_class

        choices[name] = full_name(subclass)

    return choices

def configure(full_path: str) -> None:
    if full_path in KNOWN_CONFIGS:
        return KNOWN_CONFIGS[full_path]

    parts = full_path.split(".")
    class_name = parts[-1]
    module_name = ".".join(parts[:-1])
    module = importlib.import_module(module_name)
    cla55 = getattr(module, class_name)

    if Registrable in cla55.__bases__:
        print(valid_choices(cla55))

    else:
        print(render_config(auto_config(cla55)))

# pylint: disable=protected-access,too-many-return-statements

from typing import List, TypeVar, Type, Dict
import inspect
import importlib

import torch

from allennlp.common.configuration import (
        Config, ConfigItem, NO_DEFAULT,
        _get_config_type, _docspec_comments, full_name
)
from allennlp.common.registrable import Registrable
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.training.optimizers import Optimizer as AllenNLPOptimizer
from allennlp.training.trainer import Trainer

T = TypeVar('T')

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
    defaults = [NO_DEFAULT for _ in range(num_non_default_args)] + defaults

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

def _make_item(maybe_config_item: tuple) -> ConfigItem:
    """
    Helper function to allow people to use non-named-tuples
    instead of ConfigItems.
    """
    if isinstance(maybe_config_item, ConfigItem):
        return maybe_config_item
    else:
        name, typ3, default, comment = maybe_config_item
        return ConfigItem(name, typ3, default, comment)

def configure(full_path: str = '') -> Config:
    if not full_path:
        return BASE_CONFIG

    parts = full_path.split(".")
    class_name = parts[-1]
    module_name = ".".join(parts[:-1])
    module = importlib.import_module(module_name)
    cla55 = getattr(module, class_name)
    from_params = getattr(cla55, 'from_params', None)
    if hasattr(from_params, '_config_items'):
        config_items = [_make_item(item) for item in getattr(from_params, '_config_items')]
        return Config(config_items)
    else:
        return _auto_config(cla55)



BASE_CONFIG: Config = Config([
        ConfigItem(name="dataset_reader",
                   annotation=DatasetReader,
                   default_value=NO_DEFAULT,
                   comment="specify your dataset reader here"),
        ConfigItem(name="validation_dataset_reader",
                   annotation=DatasetReader,
                   default_value=None,
                   comment="same as dataset_reader by default"),
        ConfigItem(name="train_data_path",
                   annotation=str,
                   default_value=NO_DEFAULT,
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
                   default_value=NO_DEFAULT,
                   comment="specify your model here"),
        ConfigItem(name="iterator",
                   annotation=DataIterator,
                   default_value=NO_DEFAULT,
                   comment="specify your data iterator here"),
        ConfigItem(name="trainer",
                   annotation=Trainer,
                   default_value=NO_DEFAULT,
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

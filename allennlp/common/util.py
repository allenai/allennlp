"""
Various utilities that don't fit anwhere else.
"""

from itertools import zip_longest
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union
import random

import torch
import numpy
import spacy
from spacy.language import Language as SpacyModelType

from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params

JsonDict = Dict[str, Any]  # pylint: disable=invalid-name

def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.autograd.Variable):
        return sanitize(x.data)
    elif isinstance(x, torch._TensorBase):  # pylint: disable=protected-access
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    else:
        raise ValueError("cannot sanitize {} of type {}".format(x, type(x)))

def group_by_count(iterable: List[Any], count: int, default_value: Any) -> List[List[Any]]:
    """
    Takes a list and groups it into sublists of size ``count``, using ``default_value`` to pad the
    list at the end if the list is not divisable by ``count``.

    For example:
    >>> group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 0)
    [[1, 2, 3], [4, 5, 6], [7, 0, 0]]

    This is a short method, but it's complicated and hard to remember as a one-liner, so we just
    make a function out of it.
    """
    return [list(l) for l in zip_longest(*[iter(iterable)] * count, fillvalue=default_value)]


def pad_sequence_to_length(sequence: List,
                           desired_length: int,
                           default_value: Callable[[], Any] = lambda: 0,
                           padding_on_right: bool = True) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence


A = TypeVar('A')
def add_noise_to_dict_values(dictionary: Dict[A, float], noise_param: float) -> Dict[A, float]:
    """
    Returns a new dictionary with noise added to every key in ``dictionary``.  The noise is
    uniformly distributed within ``noise_param`` percent of the value for every value in the
    dictionary.
    """
    new_dict = {}
    for key, value in dictionary.items():
        noise_value = value * noise_param
        noise = random.uniform(-noise_value, noise_value)
        new_dict[key] = value + noise
    return new_dict


def namespace_match(pattern: str, namespace: str):
    """
    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == '*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


def prepare_environment(params: Union[Params, Dict[str, Any]]):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.

    Parameters
    ----------
    params: Params object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    log_pytorch_version_info()


LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}


def get_spacy_model(spacy_model_name: str, pos_tags: bool, parse: bool, ner: bool) -> Any:
    """
    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """
    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ['vectors', 'textcat']
        if not pos_tags:
            disable.append('tagger')
        if not parse:
            disable.append('parser')
        if not ner:
            disable.append('ner')
        spacy_model = spacy.load(spacy_model_name, disable=disable)
        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]

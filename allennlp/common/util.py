"""
Various utilities that don't fit anywhere else.
"""
import hashlib
import io
import pickle
from datetime import timedelta
import importlib
import json
import logging
import os
import pkgutil
import random
import sys
from contextlib import contextmanager
from itertools import islice, zip_longest
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Sequence,
)

import numpy
import spacy
import torch
import torch.distributed as dist
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType

from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params

try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None  # type: ignore

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# If you want to have start and/or end symbols for any reason in your code, we recommend you use
# these, to have a common place to import from.  Also, it's important for some edge cases in how
# data is processed for these symbols to be lowercase, not uppercase (because we have code that
# will lowercase tokens for you in some circumstances, and we need this symbol to not change in
# those cases).
START_SYMBOL = "@start@"
END_SYMBOL = "@end@"


PathType = Union[os.PathLike, str]
T = TypeVar("T")
ContextManagerFunctionReturnType = Generator[T, None, None]


def sanitize(x: Any) -> Any:
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    # Import here to avoid circular references
    from allennlp.data.tokenizers import Token

    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
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
    elif isinstance(x, numpy.bool_):
        # Numpy bool_ need to be converted to python bool.
        return bool(x)
    elif isinstance(x, (spacy.tokens.Token, Token)):
        # Tokens get sanitized to just their text.
        return x.text
    elif isinstance(x, (list, tuple, set)):
        # Lists, tuples, and sets need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, "to_json"):
        return x.to_json()
    else:
        raise ValueError(
            f"Cannot sanitize {x} of type {type(x)}. "
            "If this is your own custom class, add a `to_json(self)` method "
            "that returns a JSON-like object."
        )


def group_by_count(iterable: List[Any], count: int, default_value: Any) -> List[List[Any]]:
    """
    Takes a list and groups it into sublists of size `count`, using `default_value` to pad the
    list at the end if the list is not divisable by `count`.

    For example:

    ```
    >>> group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 0)
    [[1, 2, 3], [4, 5, 6], [7, 0, 0]]
    ```

    This is a short method, but it's complicated and hard to remember as a one-liner, so we just
    make a function out of it.
    """
    return [list(x) for x in zip_longest(*[iter(iterable)] * count, fillvalue=default_value)]


A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def pad_sequence_to_length(
    sequence: Sequence,
    desired_length: int,
    default_value: Callable[[], Any] = lambda: 0,
    padding_on_right: bool = True,
) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    # Parameters

    sequence : `List`
        A list of objects to be padded.

    desired_length : `int`
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: `Callable`, optional (default=`lambda: 0`)
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : `bool`, optional (default=`True`)
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    # Returns

    padded_sequence : `List`
    """
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


def add_noise_to_dict_values(dictionary: Dict[A, float], noise_param: float) -> Dict[A, float]:
    """
    Returns a new dictionary with noise added to every key in `dictionary`.  The noise is
    uniformly distributed within `noise_param` percent of the value for every value in the
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
    Matches a namespace pattern against a namespace string.  For example, `*tags` matches
    `passage_tags` and `question_tags` and `tokens` matches `tokens` but not
    `stemmed_tokens`.
    """
    if pattern[0] == "*" and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


def prepare_environment(params: Params):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.

    # Parameters

    params: `Params`
        A `Params` object or dict holding the json parameters.
    """
    seed = params.pop_int("random_seed", 13370)
    numpy_seed = params.pop_int("numpy_seed", 1337)
    torch_seed = params.pop_int("pytorch_seed", 133)

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


def get_spacy_model(
    spacy_model_name: str, pos_tags: bool, parse: bool, ner: bool
) -> SpacyModelType:
    """
    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """

    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat"]
        if not pos_tags:
            disable.append("tagger")
        if not parse:
            disable.append("parser")
        if not ner:
            disable.append("ner")
        try:
            spacy_model = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            logger.warning(
                f"Spacy models '{spacy_model_name}' not found.  Downloading and installing."
            )
            spacy_download(spacy_model_name)

            # Import the downloaded model module directly and load from there
            spacy_model_module = __import__(spacy_model_name)
            spacy_model = spacy_model_module.load(disable=disable)  # type: ignore

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


@contextmanager
def pushd(new_dir: PathType, verbose: bool = False) -> ContextManagerFunctionReturnType[None]:
    """
    Changes the current directory to the given path and prepends it to `sys.path`.

    This method is intended to use with `with`, so after its usage, the current directory will be
    set to the previous value.
    """
    previous_dir = os.getcwd()
    if verbose:
        logger.info(f"Changing directory to {new_dir}")  # type: ignore
    os.chdir(new_dir)
    try:
        yield
    finally:
        if verbose:
            logger.info(f"Changing directory back to {previous_dir}")
        os.chdir(previous_dir)


@contextmanager
def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Prepends the given path to `sys.path`.

    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


def import_module_and_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage)


def peak_cpu_memory() -> Dict[int, int]:
    """
    Get peak memory usage for each worker, as measured by max-resident-set size:

    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size

    Only works on OSX and Linux, otherwise the result will be 0.0 for every worker.
    """
    if resource is None or sys.platform not in ("linux", "darwin"):
        peak_bytes = 0
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # On OSX the result is in bytes.
            peak_bytes = peak
        else:
            # On Linux the result is in kilobytes.
            peak_bytes = peak * 1_024

    if is_distributed():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes])
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0]) for _ in range(world_size)]

        # If the backend is 'nccl', this means we're training on GPUs, so these tensors
        # need to be on GPU.
        if dist.get_backend() == "nccl":
            peak_bytes_tensor = peak_bytes_tensor.cuda()
            gather_results = [x.cuda() for x in gather_results]

        dist.all_gather(gather_results, peak_bytes_tensor)

        results_dict: Dict[int, int] = {}
        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])

        return results_dict
    else:
        return {0: peak_bytes}


def peak_gpu_memory() -> Dict[int, int]:
    """
    Get the peak GPU memory usage in bytes by device.

    # Returns

    `Dict[int, int]`
        Keys are device ids as integers.
        Values are memory usage as integers in bytes.
        Returns an empty `dict` if GPUs are not available.
    """
    if not torch.cuda.is_available():
        return {}

    if is_distributed():
        # If the backend is not 'nccl', we're training on CPU.
        if dist.get_backend() != "nccl":
            return {}

        device = torch.cuda.current_device()
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes], device=device)
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0], device=device) for _ in range(world_size)]

        dist.all_gather(gather_results, peak_bytes_tensor)

        results_dict: Dict[int, int] = {}
        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])

        return results_dict
    else:
        return {0: torch.cuda.max_memory_allocated()}


def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)


def is_lazy(iterable: Iterable[A]) -> bool:
    """
    Checks if the given iterable is lazy,
    which here just means it's not a list.
    """
    return not isinstance(iterable, list)


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def log_frozen_and_tunable_parameter_names(model: torch.nn.Module) -> None:
    frozen_parameter_names, tunable_parameter_names = get_frozen_and_tunable_parameter_names(model)

    logger.info("The following parameters are Frozen (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)

    logger.info("The following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)


def get_frozen_and_tunable_parameter_names(
    model: torch.nn.Module,
) -> Tuple[Iterable[str], Iterable[str]]:
    frozen_parameter_names = (
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    )
    tunable_parameter_names = (
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    return frozen_parameter_names, tunable_parameter_names


def dump_metrics(file_path: Optional[str], metrics: Dict[str, Any], log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    if file_path:
        with open(file_path, "w") as metrics_file:
            metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


def flatten_filename(file_path: str) -> str:
    return file_path.replace("/", "_SLASH_")


def is_distributed() -> bool:
    """
    Checks if the distributed process group is available and has been initialized
    """
    return dist.is_available() and dist.is_initialized()


def sanitize_wordpiece(wordpiece: str) -> str:
    """
    Sanitizes wordpieces from BERT, RoBERTa or ALBERT tokenizers.
    """
    if wordpiece.startswith("##"):
        return wordpiece[2:]
    elif wordpiece.startswith("Ġ"):
        return wordpiece[1:]
    elif wordpiece.startswith("▁"):
        return wordpiece[1:]
    else:
        return wordpiece


def sanitize_ptb_tokenized_string(text: str) -> str:
    """
    Sanitizes string that was tokenized using PTBTokenizer
    """
    tokens = text.split(" ")
    if len(tokens) == 0:
        return text

    # Replace quotation marks and parentheses
    token_map = {
        "``": '"',
        "''": '"',
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
        "<s>": "",
        "</s>": "",
    }

    # Merge punctuation with previous tokens
    punct_forward = {"`", "$", "#"}
    punct_backward = {".", ",", "!", "?", ":", ";", "%", "'"}

    # Exact matches that get merged forward or backward
    em_forward = {"(", "[", "{"}
    em_backward = {"n't", "na", ")", "]", "}"}

    new_tokens: List[str] = []

    merge_fwd = False
    for i, orig_token in enumerate(tokens):
        tokens[i] = token_map[orig_token.lower()] if orig_token.lower() in token_map else orig_token
        new_token = tokens[i].lower()

        # merge_fwd was set by previous token, so it should be prepended to current token
        if merge_fwd:
            tokens[i] = tokens[i - 1] + tokens[i]

        if len(tokens[i]) == 0:
            continue

        # Special cases for `` and '', those tells us if " is the start or end of a quotation.
        # Also always merge tokens starting with ' backward and don't merge back if we just merged forward
        merge_bckwd = not merge_fwd and (
            orig_token == "''"
            or new_token in em_backward
            or new_token.startswith("'")
            or all(c in punct_backward for c in new_token)
        )
        merge_fwd = (
            orig_token == "``"
            or new_token in em_forward
            or all(c in punct_forward for c in new_token)
        )

        if merge_bckwd and new_tokens:
            new_tokens[-1] += tokens[i]
        elif not new_tokens or not merge_fwd or i == len(tokens) - 1:
            new_tokens.append(tokens[i])

    return " ".join(new_tokens)


def find_open_port() -> int:
    """
    Find a random open port on local host.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Passes 0 means find any open port.
        # See https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        sock.bind(("", 0))
        return sock.getsockname()[1]


def format_timedelta(td: timedelta) -> str:
    """
    Format a timedelta for humans.
    """
    if td.days > 1:
        return f"{td.days} days"
    elif td.days > 0:
        return f"{td.days} day"
    else:
        hours, remainder = divmod(td.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 1:
            return f"{hours} hours"
        elif hours > 0:
            return f"{hours} hour, {minutes} mins"
        else:
            return f"{minutes} mins"


def format_size(size: int) -> str:
    """
    Format a size (in bytes) for humans.
    """
    GBs = size / (1024 * 1024 * 1024)
    if GBs >= 10:
        return f"{int(round(GBs, 0))}G"
    if GBs >= 1:
        return f"{round(GBs, 1):.1f}G"
    MBs = size / (1024 * 1024)
    if MBs >= 10:
        return f"{int(round(MBs, 0))}M"
    if MBs >= 1:
        return f"{round(MBs, 1):.1f}M"
    KBs = size / 1024
    if KBs >= 10:
        return f"{int(round(KBs, 0))}K"
    if KBs >= 1:
        return f"{round(KBs, 1):.1f}K"
    return f"{size}B"


def nan_safe_tensor_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result


def shuffle_iterable(i: Iterable[T], pool_size: int = 1024) -> Iterable[T]:
    import random

    i = iter(i)
    pool = []

    # fill up the pool
    for item in i:
        pool.append(item)
        if len(pool) >= pool_size:
            break

    # play in it
    while len(pool) > 0:
        index = random.randrange(len(pool))
        yield pool[index]
        try:
            pool[index] = next(i)
        except StopIteration:
            del pool[index]
            break

    # drain it
    random.shuffle(pool)
    yield from pool


def cycle_iterator_function(iterator_function: Callable[[], Iterable[T]]) -> Iterator[T]:
    """
    Functionally equivalent to `itertools.cycle(iterator_function())`, but this function does not
    cache the result of calling the iterator like `cycle` does.  Instead, we just call
    `iterator_function()` again whenever we get a `StopIteration`.  This should only be preferred
    over `itertools.cycle` in cases where you're sure you don't want the caching behavior that's
    done in `itertools.cycle`.
    """
    iterator = iter(iterator_function())
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterator_function())


def hash_object(o: Any) -> str:
    """Returns a 32-character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        pickle.dump(o, buffer)
        m.update(buffer.getbuffer())
        return m.hexdigest()

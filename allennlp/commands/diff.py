"""
# Examples

```bash
allennlp diff \
    https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin \
    https://storage.googleapis.com/allennlp-public-models/transformer-qa-2020-10-03.tar.gz!weights.th \
    --strip-prefix-1 'roberta.' \
    --strip-prefix-2 '_text_field_embedder.token_embedder_tokens.transformer_model.'
```
"""
from collections import OrderedDict
import argparse
import logging
from os import PathLike
import re
from typing import Union, Optional, Dict, List, Tuple, NamedTuple, cast

from overrides import overrides
import termcolor
import torch

from allennlp.commands.subcommand import Subcommand
from allennlp.common.file_utils import cached_path


logger = logging.getLogger(__name__)


@Subcommand.register("diff")
class Diff(Subcommand):
    requires_plugins: bool = False

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Display a diff between two model checkpoints."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help=description,
        )
        subparser.set_defaults(func=_diff)
        subparser.add_argument(
            "checkpoint1",
            type=str,
            help="""The URL or path to the first PyTorch checkpoint file (e.g. '.pt' or '.bin').""",
        )
        subparser.add_argument(
            "checkpoint2",
            type=str,
            help="""The URL or path to the second PyTorch checkpoint file.""",
        )
        subparser.add_argument(
            "--strip-prefix-1",
            type=str,
            help="""A prefix to remove from all of the first checkpoint's keys.""",
        )
        subparser.add_argument(
            "--strip-prefix-2",
            type=str,
            help="""A prefix to remove from all of the second checkpoint's keys.""",
        )
        return subparser


def load_state_dict(
    path: Union[PathLike, str],
    strip_prefix: Optional[str] = None,
    ignore: Optional[List[str]] = None,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load a PyTorch model state dictionary from a checkpoint at the given `path`.

    # Parameters

    path : `Union[PathLike, str]`, required

    strip_prefix : `Optional[str]`, optional (default = `None`)
        A prefix to remove from all of the state dict keys.

    ignore : `Optional[List[str]]`, optional (default = `None`)
        Optional list of regular expressions. Keys that match any of these will be removed
        from the state dict.

        !!! Note
            If `strip_prefix` is given, the regular expressions in `ignore` are matched
            before the prefix is stripped.

    strict : `bool`, optional (default = `True`)
        If `True` (the default) and `strip_prefix` was never used or any of the regular expressions
        in `ignore` never matched, a `ValueError` will be raised.

    # Returns

    `Dict[str, torch.Tensor]`
        An ordered dictionary of the state.
    """
    state = torch.load(path, map_location="cpu")
    out: Dict[str, torch.Tensor] = OrderedDict()

    if ignore is not None and not isinstance(ignore, list):
        # If user accidentally passed in something that is not a list - like a string,
        # which is easy to do - the user would be confused why the resulting state dict
        # is empty.
        raise ValueError("'ignore' parameter should be a list")

    # In 'strict' mode, we need to keep track of whether we've used `strip_prefix`
    # and which regular expressions in `ignore` we've used.
    strip_prefix_used: Optional[bool] = None
    ignore_used: Optional[List[bool]] = None
    if strict and strip_prefix is not None:
        strip_prefix_used = False
    if strict and ignore:
        ignore_used = [False] * len(ignore)

    for key in state.keys():
        ignore_key = False
        if ignore:
            for i, pattern in enumerate(ignore):
                if re.match(pattern, key):
                    if ignore_used:
                        ignore_used[i] = True
                    logger.warning("ignoring %s from state dict", key)
                    ignore_key = True
                    break

        if ignore_key:
            continue

        new_key = key

        if strip_prefix and key.startswith(strip_prefix):
            strip_prefix_used = True
            new_key = key[len(strip_prefix) :]
            if not new_key:
                raise ValueError("'strip_prefix' resulted in an empty string for a key")

        out[new_key] = state[key]

    if strip_prefix_used is False:
        raise ValueError(f"'strip_prefix' of '{strip_prefix}' was never used")
    if ignore is not None and ignore_used is not None:
        for pattern, used in zip(ignore, ignore_used):
            if not used:
                raise ValueError(f"'ignore' pattern '{pattern}' didn't have any matches")

    return out


class Keep(NamedTuple):
    key: str
    shape: Tuple[int, ...]

    def display(self):
        termcolor.cprint(f" {self.key}, shape = {self.shape}")


class Insert(NamedTuple):
    key: str
    shape: Tuple[int, ...]

    def display(self):
        termcolor.cprint(f"+{self.key}, shape = {self.shape}", "green")


class Remove(NamedTuple):
    key: str
    shape: Tuple[int, ...]

    def display(self):
        termcolor.cprint(f"-{self.key}, shape = {self.shape}", "red")


class Modify(NamedTuple):
    key: str
    shape: Tuple[int, ...]
    distance: float

    def display(self):
        termcolor.cprint(f"!{self.key}, shape = {self.shape}, △ = {self.distance:.4f}", "yellow")


class _Frontier(NamedTuple):
    x: int
    history: List[Union[Keep, Insert, Remove]]


def _finalize(
    history: List[Union[Keep, Insert, Remove]],
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
) -> List[Union[Keep, Insert, Remove, Modify]]:
    out = cast(List[Union[Keep, Insert, Remove, Modify]], history)
    for i, step in enumerate(out):
        if isinstance(step, Keep):
            a_tensor = state_dict_a[step.key]
            b_tensor = state_dict_b[step.key]
            dist = a_tensor.dist(b_tensor).item()
            if dist != 0.0:
                out[i] = Modify(step.key, step.shape, dist)
    return out


def checkpoint_diff(
    state_dict_a: Dict[str, torch.Tensor], state_dict_b: Dict[str, torch.Tensor]
) -> List[Union[Keep, Insert, Remove, Modify]]:
    """
    Uses a modified version of the Myers diff algorithm to compute a representation
    of the diff between two model state dictionaries.

    The only difference is that in addition to the `Keep`, `Insert`, and `Remove`
    operations, we add `Modify`. This corresponds to keeping a parameter
    but changing its weights (not the shape).

    Adapted from [this gist]
    (https://gist.github.com/adamnew123456/37923cf53f51d6b9af32a539cdfa7cc4).
    """
    param_list_a = [(k, tuple(v.shape)) for k, v in state_dict_a.items()]
    param_list_b = [(k, tuple(v.shape)) for k, v in state_dict_b.items()]

    # This marks the farthest-right point along each diagonal in the edit
    # graph, along with the history that got it there
    frontier: Dict[int, _Frontier] = {1: _Frontier(0, [])}

    def one(idx):
        """
        The algorithm Myers presents is 1-indexed; since Python isn't, we
        need a conversion.
        """
        return idx - 1

    a_max = len(param_list_a)
    b_max = len(param_list_b)
    for d in range(0, a_max + b_max + 1):
        for k in range(-d, d + 1, 2):
            # This determines whether our next search point will be going down
            # in the edit graph, or to the right.
            #
            # The intuition for this is that we should go down if we're on the
            # left edge (k == -d) to make sure that the left edge is fully
            # explored.
            #
            # If we aren't on the top (k != d), then only go down if going down
            # would take us to territory that hasn't sufficiently been explored
            # yet.
            go_down = k == -d or (k != d and frontier[k - 1].x < frontier[k + 1].x)

            # Figure out the starting point of this iteration. The diagonal
            # offsets come from the geometry of the edit grid - if you're going
            # down, your diagonal is lower, and if you're going right, your
            # diagonal is higher.
            if go_down:
                old_x, history = frontier[k + 1]
                x = old_x
            else:
                old_x, history = frontier[k - 1]
                x = old_x + 1

            # We want to avoid modifying the old history, since some other step
            # may decide to use it.
            history = history[:]
            y = x - k

            # We start at the invalid point (0, 0) - we should only start building
            # up history when we move off of it.
            if 1 <= y <= b_max and go_down:
                history.append(Insert(*param_list_b[one(y)]))
            elif 1 <= x <= a_max:
                history.append(Remove(*param_list_a[one(x)]))

            # Chew up as many diagonal moves as we can - these correspond to common lines,
            # and they're considered "free" by the algorithm because we want to maximize
            # the number of these in the output.
            while x < a_max and y < b_max and param_list_a[one(x + 1)] == param_list_b[one(y + 1)]:
                x += 1
                y += 1
                history.append(Keep(*param_list_a[one(x)]))

            if x >= a_max and y >= b_max:
                # If we're here, then we've traversed through the bottom-left corner,
                # and are done.
                return _finalize(history, state_dict_a, state_dict_b)
            else:
                frontier[k] = _Frontier(x, history)

    assert False, "Could not find edit script"


def _diff(args: argparse.Namespace):
    checkpoint_1_path = cached_path(args.checkpoint1, extract_archive=True)
    checkpoint_2_path = cached_path(args.checkpoint2, extract_archive=True)
    checkpoint_1 = load_state_dict(
        checkpoint_1_path, strip_prefix=args.strip_prefix_1, strict=False
    )
    checkpoint_2 = load_state_dict(
        checkpoint_2_path, strip_prefix=args.strip_prefix_2, strict=False
    )
    for step in checkpoint_diff(checkpoint_1, checkpoint_2):
        step.display()

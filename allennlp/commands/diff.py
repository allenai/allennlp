"""
# Examples

```bash
allennlp diff \
    hf://roberta-large/pytorch_model.bin \
    https://storage.googleapis.com/allennlp-public-models/transformer-qa-2020-10-03.tar.gz \
    --strip-prefix-1 'roberta.' \
    --strip-prefix-2 '_text_field_embedder.token_embedder_tokens.transformer_model.'
```
"""
import argparse
import logging
from typing import Union, Dict, List, Tuple, NamedTuple, cast

from overrides import overrides
import termcolor
import torch

from allennlp.commands.subcommand import Subcommand
from allennlp.common.file_utils import cached_path
from allennlp.nn.util import read_state_dict


logger = logging.getLogger(__name__)


@Subcommand.register("diff")
class Diff(Subcommand):
    requires_plugins: bool = False

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Display a diff between two model checkpoints."""
        long_description = (
            description
            + """
        In the output, lines start with either a "+", "-", "!", or empty space " ".
        "+" means the corresponding parameter is present in the 2nd checkpoint but not the 1st.
        "-" means the corresponding parameter is present in the 1st checkpoint but not the 2nd.
        "!" means the corresponding parameter is present in both, but has different weights (same shape)
        according to the distance calculation and the '--threshold' value.
        And " " means the corresponding parameter is considered identical in both, i.e.
        the distance falls below the threshold.
        The distance between two tensors is calculated as the root of the
        mean squared difference, multiplied by the '--scale' parameter.
        """
        )
        subparser = parser.add_parser(
            self.name,
            description=long_description,
            help=description,
        )
        subparser.set_defaults(func=_diff)
        subparser.add_argument(
            "checkpoint1",
            type=str,
            help="""the URL, path, or other identifier (see '--checkpoint-type-1')
            to the 1st PyTorch checkpoint.""",
        )
        subparser.add_argument(
            "checkpoint2",
            type=str,
            help="""the URL, path, or other identifier (see '--checkpoint-type-2')
            to the 2nd PyTorch checkpoint.""",
        )
        subparser.add_argument(
            "--strip-prefix-1",
            type=str,
            help="""a prefix to remove from all of the 1st checkpoint's keys.""",
        )
        subparser.add_argument(
            "--strip-prefix-2",
            type=str,
            help="""a prefix to remove from all of the 2nd checkpoint's keys.""",
        )
        subparser.add_argument(
            "--scale",
            type=float,
            default=1.0,
            help="""controls the scale of the distance calculation.""",
        )
        subparser.add_argument(
            "--threshold",
            type=float,
            default=1e-5,
            help="""the threshold for the distance between two tensors,
            under which the two tensors are considered identical.""",
        )
        return subparser


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
        termcolor.cprint(
            f"!{self.key}, shape = {self.shape}, distance = {self.distance:.4f}", "yellow"
        )


class _Frontier(NamedTuple):
    x: int
    history: List[Union[Keep, Insert, Remove]]


def _finalize(
    history: List[Union[Keep, Insert, Remove]],
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    scale: float,
    threshold: float,
) -> List[Union[Keep, Insert, Remove, Modify]]:
    out = cast(List[Union[Keep, Insert, Remove, Modify]], history)
    for i, step in enumerate(out):
        if isinstance(step, Keep):
            a_tensor = state_dict_a[step.key]
            b_tensor = state_dict_b[step.key]
            with torch.no_grad():
                dist = (scale * torch.nn.functional.mse_loss(a_tensor, b_tensor).sqrt()).item()
            if dist > threshold:
                out[i] = Modify(step.key, step.shape, dist)
    return out


def checkpoint_diff(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    scale: float,
    threshold: float,
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
                return _finalize(history, state_dict_a, state_dict_b, scale, threshold)
            else:
                frontier[k] = _Frontier(x, history)

    assert False, "Could not find edit script"


def _get_checkpoint_path(checkpoint: str) -> str:
    if checkpoint.endswith(".tar.gz"):
        return cached_path(checkpoint + "!weights.th", extract_archive=True)
    elif ".tar.gz!" in checkpoint:
        return cached_path(checkpoint, extract_archive=True)
    else:
        return cached_path(checkpoint)


def _diff(args: argparse.Namespace):
    checkpoint_1_path = _get_checkpoint_path(args.checkpoint1)
    checkpoint_2_path = _get_checkpoint_path(args.checkpoint2)
    checkpoint_1 = read_state_dict(
        checkpoint_1_path, strip_prefix=args.strip_prefix_1, strict=False
    )
    checkpoint_2 = read_state_dict(
        checkpoint_2_path, strip_prefix=args.strip_prefix_2, strict=False
    )
    for step in checkpoint_diff(checkpoint_1, checkpoint_2, args.scale, args.threshold):
        step.display()

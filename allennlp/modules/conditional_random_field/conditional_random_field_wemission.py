"""
Conditional random field with emission-based weighting
"""
from typing import List, Tuple

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.conditional_random_field.conditional_random_field import (
    ConditionalRandomField,
)


class ConditionalRandomFieldWeightEmission(ConditionalRandomField):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    This is a weighted version of `ConditionalRandomField` which accepts a `label_weights`
    parameter to be used in the loss function in order to give different weights for each
    token depending on its label. The method implemented here is based on the simple idea
    of weighting emission scores using the weight given for the corresponding tag.

    There are two other sample weighting methods implemented. You can find more details
    about them in: https://eraldoluis.github.io/2022/05/10/weighted-crf.html

    # Parameters

    num_tags : `int`, required
        The number of tags.
    label_weights : `List[float]`, required
        A list of weights to be used in the loss function in order to
        give different weights for each token depending on its label.
        `len(label_weights)` must be equal to `num_tags`. This is useful to
        deal with highly unbalanced datasets. The method implemented here is
        based on the simple idea of weighting emission scores using the weight
        given for the corresponding tag.
    constraints : `List[Tuple[int, int]]`, optional (default = `None`)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default = `True`)
        Whether to include the start and end transition parameters.
    """

    def __init__(
        self,
        num_tags: int,
        label_weights: List[float],
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__(num_tags, constraints, include_start_end_transitions)

        if label_weights is None:
            raise ConfigurationError("label_weights must be given")

        self.register_buffer("label_weights", torch.Tensor(label_weights))

    def forward(
        self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """Computes the log likelihood for the given batch of input sequences $(x,y)$

        Args:
            inputs (torch.Tensor): (batch_size, sequence_length, num_tags) tensor of logits for the inputs $x$
            tags (torch.Tensor): (batch_size, sequence_length) tensor of tags $y$
            mask (torch.BoolTensor, optional): (batch_size, sequence_length) tensor of masking flags.
                Defaults to None.

        Returns:
            torch.Tensor: (batch_size,) log likelihoods $log P(y|x)$ for each input
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = mask.to(torch.bool)

        label_weights = self.label_weights

        # scale the logits for all examples and all time steps
        inputs = inputs * label_weights.view(1, 1, -1)

        log_denominator = self._input_likelihood(inputs, self.transitions, mask)
        log_numerator = self._joint_likelihood(inputs, self.transitions, tags, mask)

        return torch.sum(log_numerator - log_denominator)

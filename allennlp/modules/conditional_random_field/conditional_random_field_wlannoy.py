"""
Conditional random field with weighting based on Lannoy et al. (2019) approach
"""
from typing import List, Tuple

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.conditional_random_field.conditional_random_field import (
    ConditionalRandomField,
)


class ConditionalRandomFieldWeightLannoy(ConditionalRandomField):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    This is a weighted version of `ConditionalRandomField` which accepts a `label_weights`
    parameter to be used in the loss function in order to give different weights for each
    token depending on its label. The method implemented here is based on the paper
    *Weighted conditional random fields for supervised interpatient heartbeat
    classification* proposed by De Lannoy et. al (2019).
    See https://perso.uclouvain.be/michel.verleysen/papers/ieeetbe12gdl.pdf for more details.

    There are two other sample weighting methods implemented. You can find more details
    about them in: https://eraldoluis.github.io/2022/05/10/weighted-crf.html

    # Parameters

    num_tags : `int`, required
        The number of tags.
    label_weights : `List[float]`, required
        A list of weights to be used in the loss function in order to
        give different weights for each token depending on its label.
        `len(label_weights)` must be equal to `num_tags`. This is useful to
        deal with highly unbalanced datasets. The method implemented here was based on
        the paper *Weighted conditional random fields for supervised interpatient heartbeat
        classification* proposed by De Lannoy et. al (2019).
        See https://perso.uclouvain.be/michel.verleysen/papers/ieeetbe12gdl.pdf
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

        log_denominator = self._input_likelihood_lannoy(inputs, tags, mask)
        log_numerator = self._joint_likelihood_lannoy(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def _input_likelihood_lannoy(
        self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.

        Compute this value using the scaling trick instead of the log domain trick, since
        this is necessary to implement the label-weighting method by Lannoy et al. (2012).
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # insert an 1-sized second dimension to match z.shape
        label_weights = self.label_weights.view(num_tags, 1)

        # emit_scores.shape = (batch_size, num_tags)
        emit_scores = logits[0]

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        # alpha.shape = (batch_size, num_tags)
        if self.include_start_end_transitions:
            alpha = torch.exp(self.start_transitions.view(1, num_tags) + emit_scores)
        else:
            alpha = torch.exp(emit_scores)

        # z.shape = (batch_size, 1)
        z = alpha.sum(dim=1, keepdim=True)
        alpha = alpha / z
        sum_log_z = torch.log(z) * label_weights[tags[0]]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # multiply the logits by the label weights
            # logits[i].shape: (batch_size, num_tags)
            # emit_scores = torch.mul(logits[i], label_weights)
            emit_scores = logits[i]

            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = emit_scores.view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag (i-1), so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha * torch.exp(emit_scores + transition_scores)

            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            alpha = inner.sum(dim=1) * mask[i].view(batch_size, 1) + alpha * (~mask[i]).view(
                batch_size, 1
            )

            # scale alphas to avoid underflow (sum of alphas equal to 1)
            z = alpha.sum(dim=1, keepdim=True)
            alpha = alpha / z
            # weight z (normalization factor) according to the current tag
            sum_log_z += torch.log(z) * label_weights[tags[i]]

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            alpha = alpha * torch.exp(self.end_transitions.view(1, num_tags))
            z = alpha.sum(dim=1, keepdim=True)
            # alpha = alpha / z  # this step is unnecessary since alpha is not used anymore
            sum_log_z += torch.log(z)

        return sum_log_z.squeeze(1)

    def _joint_likelihood_lannoy(
        self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        label_weights = self.label_weights

        # weight transition score using current_tag, i.e., t(i,j) will be t(i,j)*w(i),
        # where t(i,j) is the score to transition from i to j and w(i) is the weight
        # for tag i.
        transitions = self.transitions * label_weights.view(-1, 1)

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Weight emit scores by label.
            emit_score *= label_weights[current_tag.view(-1)]

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        # Weight last emit scores by label weights.
        last_input_score = last_input_score * label_weights[last_tags.view(-1)]

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

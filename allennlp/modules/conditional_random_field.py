"""
Conditional random field
"""
from typing import List, Tuple, Dict

import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util


def allowed_transitions(constraint_type: str, tokens: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given tokens and a constraint type, returns the allowed transitions.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are "BIO" and "BIOUL".
    tokens : ``Dict[int, str]``, required
        A mapping {token_id -> token}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_token_id, to_token_id)
    """
    allowed = []
    if constraint_type == "BIOUL":
        for i, (from_bioul, *from_entity) in tokens.items():
            for j, (to_bioul, *to_entity) in tokens.items():

                is_allowed = any([
                        # O can transition to O, B-* or U-*
                        # L-x can transition to O, B-*, or U-*
                        # U-x can transition to O, B-*, or U-*
                        from_bioul in ('O', 'L', 'U') and to_bioul in ('O', 'B', 'U'),
                        # B-x can only transition to I-x or L-x
                        # I-x can only transition to I-x or L-x
                        from_bioul in ('B', 'I') and to_bioul in ('I', 'L') and from_entity == to_entity
                ])

                if is_allowed:
                    allowed.append((i, j))

    elif constraint_type == "BIO":
        for i, (from_bio, *from_entity) in tokens.items():
            for j, (to_bio, *to_entity) in tokens.items():

                is_allowed = any([
                        # Can always transition to O or B-x
                        to_bio in ('O', 'B'),
                        # Can only transition to I-x from B-x or I-x
                        to_bio == 'I' and from_bio in ('B', 'I') and from_entity == to_entity
                ])

                if is_allowed:
                    allowed.append((i, j))

    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")

    return allowed


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    constraints : List[Tuple[int, int]], optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
    """
    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        if constraints is None:
            self._constraint_mask = None
        else:
            constraint_mask = torch.Tensor(num_tags, num_tags).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.

            self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal(self.transitions)
        torch.nn.init.normal(self.start_transitions)
        torch.nn.init.normal(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        alpha = self.start_transitions.view(1, num_tags) + logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            alpha = (util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        stops = alpha + self.end_transitions.view(1, num_tags)

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return util.logsumexp(stops)

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, num_tags = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        score = self.start_transitions.index_select(0, tags[0])

        # Broadcast the transition scores to one per batch element
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags)

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i+1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = (
                    broadcast_transitions
                    # Choose the current_tag-th row for each input
                    .gather(1, current_tag.view(batch_size, 1, 1).expand(batch_size, 1, num_tags))
                    # Squeeze down to (batch_size, num_tags)
                    .squeeze(1)
                    # Then choose the next_tag-th column for each of those
                    .gather(1, next_tag.view(batch_size, 1))
                    # And squeeze down to (batch_size,)
                    .squeeze(1)
            )

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size).expand(sequence_length, batch_size))

        # Is (sequence_length, batch_size), but all the columns are the same, so take the first.
        last_tags = last_tags[0]

        # Compute score of transitioning to `stop_tag` from each "last tag".
        last_transition_score = self.end_transitions.index_select(0, last_tags)

        # Add the last input if it's not masked.
        last_inputs = logits[-1]                                         # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        """
        Computes the log likelihood.
        """
        # pylint: disable=arguments-differ
        if mask is None:
            mask = torch.autograd.Variable(torch.ones(*tags.size()).long())

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: Variable, mask: Variable) -> List[List[int]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        """
        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        if self._constraint_mask is None:
            constrained_transitions = self.transitions
        else:
            constrained_transitions = (self.transitions * self._constraint_mask +
                                       -10000.0 * (1 - self._constraint_mask))

        transitions[:num_tags, :num_tags] = constrained_transitions.data
        transitions[start_tag, :num_tags] = self.start_transitions.data
        transitions[:num_tags, end_tag] = self.end_transitions.data

        all_tags = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, _ = util.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and append.
            all_tags.append(viterbi_path[1:-1])

        return all_tags

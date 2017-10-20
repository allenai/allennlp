"""
Conditional random field
"""
import torch

from allennlp.nn.util import logsumexp

class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    start_tag_id : int, required
        The id of the sentinel <START> tag.
    stop_tag_id : int, required
        The id of the sentinel <STOP> tag.
    """
    def __init__(self,
                 num_tags: int,
                 start_tag_id: int,
                 stop_tag_id: int) -> None:
        super().__init__()

        self.num_tags = num_tags
        self.start_tag_id = start_tag_id
        self.stop_tag_id = stop_tag_id

        # transitions[i, j] is the logit for transitioning from state i to state j
        self.transitions = torch.nn.Parameter(torch.randn(num_tags, num_tags))

        # We never transition to the start tag and we never transition from the stop tag.
        self.transitions.data[:, start_tag_id] = -10000
        self.transitions.data[stop_tag_id, :] = -10000

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
        alpha = self.transitions[self.start_tag_id].view(1, num_tags) + logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, prev_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the prev_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (prev_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the prev_tag, so we broadcast along the next_tag axis.
            alpha = alpha.view(batch_size, num_tags, 1)

            # We add the emit + transition scores only if timestep i - 1 was included.
            inner = alpha + (emit_scores + transition_scores) * mask[i - 1].view(batch_size, 1, 1)

            # Now we logsumexp over the "prev_tag" axis.
            alpha = logsumexp(inner, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        stops = alpha + self.transitions[:, self.stop_tag_id].contiguous().view(1, num_tags)

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

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
        score = self.transitions[self.start_tag_id].index_select(0, tags[0])

        # Broadcast the transition scores to one per batch element
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags)

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i+1]

            # The scores for transitioning from prev_tag to next_tag
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

            # The score for using prev_tag
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
        last_transition_score = self.transitions.index_select(0, last_tags)[:, self.stop_tag_id]

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

"""
Conditional random field
"""
import torch

from allennlp.nn.util import log_sum_exp

class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    start_tag : int, required
        The id of the sentinel <START> tag.
    stop_tag : int, required
        The id of the sentinel <STOP> tag.
    """
    def __init__(self,
                 num_tags: int,
                 start_tag: int,
                 stop_tag: int) -> None:
        super().__init__()

        self.num_tags = num_tags
        self.start_tag = start_tag
        self.stop_tag = stop_tag

        # transitions[i, j] is the logit for transitioning to state i from state j
        self.transitions = torch.nn.Parameter(torch.randn(num_tags, num_tags))

        # We never transition to the start tag and we never transition from the stop tag.
        self.transitions.data[start_tag, :] = -10000
        self.transitions.data[:, stop_tag] = -10000

    def _input_likelihood(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) numerator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = inputs.size()
        mask = mask.float()
        # masks[i] is the (batch_size,) mask tensor for timestep i
        masks = [mask[:, i].contiguous() for i in range(sequence_length)]

        # We will step through time and accumulate the likelihood at each step.
        # Note that we will accumulate a (batch_size, num_tags) tensor of likelihoods
        # since the likelihood of each step is

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        alpha = self.transitions[:, self.start_tag].contiguous().view(1, num_tags) + inputs[:, 0, :]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, next_tag, prev_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the prev_tag axis.
            emit_scores = inputs[:, i].contiguous().view(batch_size, num_tags, 1)
            # Transition scores are (next_tag, prev_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the prev_tag, so we broadcast along the next_tag axis.
            alpha = alpha.view(batch_size, 1, num_tags)

            # We add the emit+transition scores only if timestep i-1 was included.
            inner = alpha + (emit_scores + transition_scores) * masks[i-1].view(batch_size, 1, 1)

            # Now we log_sum_exp over the "prev_tag" axis.
            alpha = log_sum_exp(inner)

        # Every sequence needs to end with a transition to the stop_tag.
        stops = alpha + self.transitions[self.stop_tag].view(1, num_tags)

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return log_sum_exp(stops)

    def _joint_likelihood(self,
                          inputs: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.ByteTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, num_tags = inputs.data.shape
        mask = mask.float()
        masks = [mask[:, i] for i in range(sequence_length)]
        # contiguous tags
        ctags = [tags[:, i].contiguous() for i in range(sequence_length)]

        # Start with the transition scores from start_tag to the first tag in each input
        score = self.transitions.index_select(0, tags[:, 0])[:, self.start_tag]

        # Broadcast the transition scores to one per batch element
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags)

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            prev_tag, next_tag = ctags[i], ctags[i+1]

            # The scores for transitioning from prev_tag to next_tag
            transition_score = (
                    broadcast_transitions
                    # Choose the next_tag-th row for each input
                    .gather(1, next_tag.view(-1, 1, 1).expand(batch_size, 1, num_tags))
                    # Squeeze down to (batch_size, num_tags)
                    .squeeze(1)
                    # Then choose the prev_tag-th column for each of those
                    .gather(1, prev_tag.view(-1, 1))
                    # And squeeze down to (batch_size,)
                    .squeeze(1)
            )

            # The score for using prev_tag
            input_score = inputs[:, i].contiguous().gather(1, prev_tag.view(-1, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * masks[i+1] + input_score * masks[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(-1).long() - 1
        last_tags = tags.gather(1, last_tag_index.view(batch_size, 1).expand(batch_size, sequence_length))

        # Is (batch_size, sequence_length), but all the columns are the same, so take the first.
        last_tags = last_tags[:, 0].contiguous()

        # Compute score of transitioning to `stop_tag` from each "last tag".
        last_transition_score = self.transitions[self.stop_tag].index_select(0, last_tags)

        # Add the last input if it's not masked.
        last_inputs = inputs[:, -1].contiguous()                         # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * masks[-1]

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
            mask = torch.autograd.Variable(torch.ones(*tags.size()).byte())

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

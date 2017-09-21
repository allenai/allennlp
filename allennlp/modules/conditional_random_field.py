"""
Conditional random field
"""

import torch

def log_sum_exp(x: torch.autograd.Variable) -> torch.autograd.Variable:  # pylint: disable=invalid-name
    """
    numerically stable log(sum(exp(x))) over only the last dimension.
    assumes x is 2-dimensional
    """
    batch_size, num_tags = x.size()

    maxes, _ = torch.max(x, -1)
    broadcast = maxes.view(batch_size, 1)
    exps = torch.exp(x - broadcast)

    return maxes + torch.log(torch.sum(exps, -1))


def log_sum_exp3(x: torch.autograd.Variable) -> torch.autograd.Variable:  # pylint: disable=invalid-name
    batch_size, num_tags, _ = x.size()

    maxes, _ = x.max(-1)
    broadcast = maxes.view(batch_size, num_tags, 1)
    exps = torch.exp(x - broadcast)
    return maxes + torch.log(torch.sum(exps, -1))



class ConditionalRandomField(torch.nn.Module):
    """
    Computes the conditional random field loss,
    which is the negative log likelihood.

    Parameters
    ----------
    num_tags: int, required
        The number of tags.
    start_tag: int, requred
        The id of the special "start" sentinel tag.
    stop_tag: int, required
        The id of the special "stop" sentinel tag.
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
        self.transitions = torch.nn.Parameter(1 * torch.randn(num_tags, num_tags))

        # We never transition to the start tag and we never transition from the stop tag
        self.transitions.data[start_tag, :] = -10000
        self.transitions.data[:, stop_tag] = -10000

    def _input_likelihood(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_tags = inputs.size()
        mask = mask.float()
        masks = [mask[:, i].contiguous() for i in range(sequence_length)]

        # alpha_0 is the transitions to the initial states, given the first word
        alpha = self.transitions[:, self.start_tag].contiguous().view(1, num_tags) + inputs[:, 0, :]

        for i in range(sequence_length - 1):
            # want to broadcast emit scores along prev_tag
            emit_scores = inputs[:, i+1].contiguous().view(batch_size, num_tags, 1)
            # want to broadcast transition scores along batch
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # want to broadcast alpha along next_tag
            alpha = alpha.view(batch_size, 1, num_tags)

            # (batch_size, next_tag, prev_tag)
            inner = (emit_scores + transition_scores) * masks[i].view(batch_size, 1, 1) + alpha

            # need to LSE over all the prev_tags
            alpha = log_sum_exp3(inner)

        # stopping
        stops = alpha + self.transitions[self.stop_tag].view(1, num_tags)

        # LSE along num_tags dim, result is (batch_size,)
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
            # Each is shape (batch_size, num_tags)
            prev_tag, next_tag = ctags[i], ctags[i+1]

            # The scores for transitioning from prev_tag to next_tag
            transition_score = (
                    broadcast_transitions
                    # Choose the next_tag-th row for each input
                    .gather(1, next_tag.view(-1, 1, 1).expand(batch_size, 1, num_tags))
                    .squeeze()
                    # And then choose the prev_tag-th column for each of those
                    .gather(1, prev_tag.view(-1, 1))
                    .squeeze()
            )

            # The score for using prev_tag
            input_score = inputs[:, i].contiguous().gather(1, prev_tag.view(-1, 1)).squeeze()

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * masks[i+1] + input_score * masks[i]

        # Transition from last state to "stop" state.
        last_transition_score = self.transitions[self.stop_tag].index_select(0, tags[:, -1])

        # Finally, add the last input if it's not masked.
        last_inputs = inputs[:, -1].contiguous()                         # (batch_size, num_tags)
        last_tags = ctags[-1]                                            # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * masks[-1]

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        """
        ``forward`` only computes the loss
        """
        # pylint: disable=arguments-differ
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

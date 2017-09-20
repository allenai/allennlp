"""
Conditional random field
"""

import torch

def log_sum_exp(x: torch.autograd.Variable) -> torch.autograd.Variable:  # pylint: disable=invalid-name
    """
    numerically stable log(sum(exp(x))) over only the last dimension.
    assumes x is 2-dimensional
    """
    batch_size, num_tags = x.data.shape

    maxes, _ = torch.max(x, -1)
    broadcast = maxes.view(batch_size, 1).expand(batch_size, num_tags)
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


    def _log_likelihood_denominator(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the denominator term for the log-likelihood calculation,
        which is

        log(sum(exp(score(inputs, state_sequence)))

        where the sum is over all possible state_sequences.
        """
        batch_size, sequence_length, num_tags = inputs.data.shape
        mask = mask.float()
        masks = [mask[:, i].contiguous() for i in range(sequence_length)]

        # At step 0, start_tag has all of the score. This ugliness gets the right device.
        init_alphas = inputs.data[0].new().resize_(batch_size, num_tags).fill_(-10000.)
        init_alphas[:, self.start_tag] = 0.

        forward_var = torch.autograd.Variable(init_alphas)

        # Iterate through the sentence
        for i in range(sequence_length):
            alphas_t = []

            # TODO(joelgrus): this could probably be vectorized
            for next_tag in range(num_tags):
                # Include emit score for the i-th tag if masks[i] is 1
                emit_score = inputs[:, i, next_tag].contiguous()
                emit_score = emit_score * masks[i]
                emit_score = emit_score.view(batch_size, 1).expand(batch_size, num_tags)

                # Include transition_score to the i-th tag if masks[i] is 1
                transition_score = self.transitions[next_tag].view(1, num_tags).expand(batch_size, num_tags)
                transition_score = transition_score * masks[i].view(batch_size, 1).expand(batch_size, num_tags)

                # Resulting score is (batch_size, num_tags)
                next_tag_var = forward_var + transition_score + emit_score

                # For this tag we log_sum_exp over all the next_tags and append as a new column in alphas_t
                alphas_t.append(log_sum_exp(next_tag_var).view(batch_size, 1))

            # At this point alphas_t is a list of num_tags (batch_size, 1) tensors
            # Concatenate to get (batch_size, num_tags)
            forward_var = torch.cat(alphas_t, 1)

        # score for stopping from each tag
        stops = self.transitions[self.stop_tag].view(1, num_tags).expand(batch_size, num_tags)
        terminal_var = forward_var + stops

        # again log_sum_exp over the tags to get a (batch_size,) output
        return log_sum_exp(terminal_var)

    def _log_likelihood_numerator(self,
                                  inputs: torch.Tensor,
                                  tags: torch.Tensor,
                                  mask: torch.ByteTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, num_tags = inputs.data.shape
        mask = mask.float()
        masks = [mask[:, i] for i in range(sequence_length)]

        # Variable to hold the numerators, shape (batch_size,), initialized to all zeros
        # This nastiness is needed to get it on the same device as the inputs (CPU or GPU).
        score = torch.autograd.Variable(inputs.data[:, 0, 0].new().resize_(batch_size).fill_(0.))

        # Add the transition scores from start_tag to the first tag in each input
        score = score + self.transitions.index_select(0, tags[:, 0])[:, self.start_tag]

        # Broadcast the transition scores to one per batch element
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags)

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            prev_tag = tags[:, i].contiguous()    # The i-th tag for each input
            next_tag = tags[:, i+1].contiguous()  # The (i+1)-th tag for each input

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
        last_tags = tags[:, -1].contiguous()                             # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * masks[-1]

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor=None) -> torch.Tensor:
        """
        ``forward`` only computes the loss
        """
        # pylint: disable=arguments-differ
        log_denominator = self._log_likelihood_denominator(inputs, mask)
        log_numerator = self._log_likelihood_numerator(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

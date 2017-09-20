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

    # (batch_size, sequence_length)
    maxes, _ = torch.max(x, -1)
    broadcast = maxes.view(batch_size, 1).expand(batch_size, num_tags)
    exps = torch.exp(x - broadcast)

    # (batch_size,)
    return maxes + torch.log(torch.sum(exps, -1))


class ConditionalRandomField(torch.nn.Module):
    """
    CRF

    Parameters
    ----------
    num_tags: int, required
        The number of tags.
    start_tag: int, requred
        The id of the special "start" sentinel tag.
    stop_tag: int, required
        The id of the special "stop" sentinel tag.

    Returns
    -------
    output: torch.Tensor
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
        batch_size, sequence_length, num_tags = inputs.data.shape

        # at step 0, start_tag has all of the score
        #init_alphas = torch.Tensor(batch_size, num_tags).fill_(-10000.)
        init_alphas = inputs.data[0].new().resize_(batch_size, num_tags).fill_(-10000.)
        init_alphas[:, self.start_tag] = 0.

        forward_var = torch.autograd.Variable(init_alphas)

        # Iterate through the sentence
        for i in range(sequence_length):
            # (batch_size,)
            mask_i = mask[:, i].float()

            alphas_t = []

            # TODO(joelgrus): vectorize this once it works
            for next_tag in range(num_tags):
                # (batch_size,) -> (batch_size, num_tags)
                emit_score = inputs[:, i, next_tag].contiguous()
                emit_score = emit_score * mask_i
                emit_score = emit_score.view(batch_size, 1).expand(batch_size, num_tags)

                # (num_tags,) -> (batch_size, num_tags)
                trans_score = self.transitions[next_tag].view(1, num_tags).expand(batch_size, num_tags)
                trans_score = trans_score * mask_i.view(batch_size, 1).expand(batch_size, num_tags)

                # (batch_size, num_tags)
                next_tag_var = forward_var + trans_score + emit_score

                # for this tag we log_sum_exp over all the next_tags
                alphas_t.append(log_sum_exp(next_tag_var).view(batch_size, 1))

            # At this point alphas_t is a list of num_tags (batch_size, 1) tensors
            # Concatenate to get (batch_size, num_tags)
            forward_var = torch.cat(alphas_t, 1)

        # (num_tags,) -> (batch_size, num_tags)
        stops = self.transitions[self.stop_tag].view(1, num_tags).expand(batch_size, num_tags)

        # (batch_size, num_tags)
        terminal_var = forward_var + stops

        # again log_sum_exp over the tags
        # (batch_size,)
        return log_sum_exp(terminal_var)

    def _log_likelihood_numerator(self,
                                  inputs: torch.Tensor,
                                  tags: torch.Tensor,
                                  mask: torch.ByteTensor) -> torch.Tensor:
        batch_size, sequence_length, num_tags = inputs.data.shape

        # Variable to hold the numerators
        #score = torch.autograd.Variable(torch.Tensor(batch_size).fill_(0.))
        score = torch.autograd.Variable(inputs.data[:, 0, 0].new().resize_(batch_size).fill_(0.))

        # Transitions from start_tag
        score = score + self.transitions.index_select(0, tags[:, 0])[:, self.start_tag]

        # Broadcast transitions
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags)

        # Actual transitions
        for i in range(sequence_length - 1):
            mask_i = mask[:, i].float()
            prev_tag = tags[:, i].contiguous()
            next_tag = tags[:, i+1].contiguous()

            transition_score = (
                    broadcast_transitions
                    .gather(1, next_tag.view(-1, 1, 1).expand(batch_size, 1, num_tags))
                    .squeeze()
                    .gather(1, prev_tag.view(-1, 1))
                    .squeeze()
            )

            input_score = inputs[:, i].contiguous().gather(1, prev_tag.view(-1, 1)).squeeze()
            score = score + transition_score * mask_i + input_score * mask_i

        # Last input and transition to stop
        last_transition_score = self.transitions[self.stop_tag].index_select(0, tags[:, -1])

        last_inputs = inputs[:, -1].contiguous()                         # (batch_size, num_tags)
        last_tags = tags[:, -1].contiguous()                             # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)
        last_mask = mask[:, -1].float()

        score = score + last_transition_score + last_input_score * last_mask

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

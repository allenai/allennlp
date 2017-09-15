"""
Conditional random field
"""

from typing import Optional, Tuple
import torch

from allennlp.common.checks import ConfigurationError

def log_sum_exp(x: torch.autograd.Variable) -> torch.autograd.Variable:
    """
    numerically stable log(sum(exp(x))) over only the last dimension.
    assumes x is 2-dimensional
    """
    batch_size, num_tags = x.shape

    # (batch_size, sequence_length)
    maxes, _ = torch.max(x, -1)
    broadcast = maxes.view(batch_size, 1).expand(batch_size, num_tags)
    exps = torch.exp(x - broadcast)

    # (batch_size,)
    return torch.log(torch.sum(exps, -1))


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
        self.transitions = torch.nn.Parameter(torch.randn(num_tags, num_tags))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[start_tag, :] = -10000
        self.transitions.data[:, stop_tag] = -10000


    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor=None):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A (batch_size, sequence_length, num_tags) Tensor.
        tags: ``torch.Tensor``, required.
            A (batch_size, sequence_length) Tensor
        mask: ``torch.Tensor``.
            A (batch_size, sequence_length) mask

        Returns
        -------
        log_likelihood
            The CRF log-likelihood of the input sequence[s]
        """
        batch_size, sequence_length, num_tags = inputs.shape

        # at step 0, start_tag has all of the score
        alphas = torch.Tensor(batch_size, 1, num_tags).fill_(-10000.)
        alphas[:, 0, self.start_tag] = 0.

        forward_var = torch.autograd.Variable(alphas)

        # Iterate through the sentence
        for i in range(sequence_length):
            alphas_t = []

            # TODO(joelgrus): vectorize this once it works
            for next_tag in range(num_tags):
                # (batch_size,)
                emit_score = inputs[:, i, next_tag]
                # (batch_size, 1)
                emit_score = emit_score.view(batch_size, 1)
                # (batch_size, num_tags)
                emit_score = emit_score.expand(batch_size, num_tags)

                # (num_tags,) probability of transitioning from each to next_tag
                trans_score = self.transitions[next_tag]
                # (1, num_tags)
                trans_score = trans_score.view(1, num_tags)
                # (batch_size, num_tags)
                trans_score = trans_score.expand(batch_size, num_tags)

                # (batch_size, num_tags)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(batch_size, 1))

            # At this point alphas_t is a list of num_tags (batch_size, 1) tensors
            # Concatenate to get (batch_size, num_tags)
            forward_var = torch.cat(alphas_t, 1)

            # Subtract off correct tag
            forward_var[:, 1, tags] -= 1

        # (num_tags,)
        stops = self.transitions[self.stop_tag]
        # (1, 1, num_tags)
        stops = stops.view(1, 1, num_tags)
        # (batch_size, 1, num_tags)
        stops = stops.expand(batch_size, 1, num_tags)

        # (batch_size, 1, num_tags)
        terminal_var = forward_var + stops

        # (batch_size, 1)
        losses = log_sum_exp(terminal_var)

        return torch.sum(losses)


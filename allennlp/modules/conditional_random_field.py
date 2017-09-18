"""
Conditional random field
"""

import torch

from allennlp.nn.util import viterbi_decode

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


    def _log_likelihood_denominator(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_tags = inputs.data.shape

        # at step 0, start_tag has all of the score
        forward_var = torch.autograd.Variable(torch.Tensor(batch_size, num_tags).fill_(-10000.))
        forward_var[:, self.start_tag] = 0.

        # Iterate through the sentence
        for i in range(sequence_length):
            alphas_t = []

            # TODO(joelgrus): vectorize this once it works
            for next_tag in range(num_tags):
                # (batch_size,) -> (batch_size, num_tags)
                emit_score = inputs[:, i, next_tag].contiguous()
                emit_score = emit_score.view(batch_size, 1).expand(batch_size, num_tags)

                # (num_tags,) -> (batch_size, num_tags)
                trans_score = self.transitions[next_tag].view(1, num_tags).expand(batch_size, num_tags)

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
                                  mask: torch.ByteTensor=None) -> torch.Tensor:
        batch_size, sequence_length, num_tags = inputs.data.shape

        # Variable to hold the numerators
        score = torch.autograd.Variable(torch.Tensor(batch_size).fill_(0.))

        # Initial transitions:
        # TODO(joelgrus) vectorize
        for j in range(batch_size):
            prev_tag, next_tag = self.start_tag, tags[j, 0]
            score[j] = score[j] + self.transitions.index_select(0, next_tag)[0, prev_tag]

        for i in range(sequence_length - 1):
            # TODO(joelgrus) vectorize
            for j in range(batch_size):
                if mask is None or mask[j, i].data[0]:
                    prev_tag, next_tag = tags[j, i], tags[j, i+1]
                    trans = self.transitions.index_select(0, next_tag).index_select(1, prev_tag)
                    inp = inputs[j, i].index_select(0, next_tag)
                    score[j] = score[j] + trans + inp

        # and add the last transition too
        # TODO(joelgrus) vectorize
        for j in range(batch_size):
            prev_tag, next_tag = tags[j, -1], self.stop_tag
            trans = self.transitions[next_tag].index_select(0, prev_tag)
            score[j] = score[j] + trans

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor=None) -> torch.Tensor:
        """
        ``forward`` only computes the loss
        """
        # pylint: disable=arguments-differ
        log_denominator = self._log_likelihood_denominator(inputs)
        log_numerator = self._log_likelihood_numerator(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

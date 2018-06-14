import logging

import torch
from torch import nn

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

from overrides import overrides

from math import sqrt

logger = logging.getLogger(__name__)

class AdaptiveSoftmax(nn.Module):
    """
    The implementation of the adaptive softmax, which is a two layer hierarchical
    softmax, while the hierarchical is constructed based on the word frequency.
    This implementation is based on
    <https://github.com/facebookresearch/adaptive-softmax> and
    <https://github.com/rosinality/adaptive-softmax-pytorch>.

    Parameters
    ----------
    input_dim : ``int``, required
        The dimension of the inputs to the adaptive softmax
    vocab : ``Vocabulary``, required
        The Vocabulary used for the adaptive softmax.
    cutoff : ``list``, required
        The cutoff size of the second layer of the adaptive softmax

    """

    def __init__(self, input_dim: int, vocab: Vocabulary, cutoff: list) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.cutoff = cutoff
        self.vocab = vocab
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Linear(input_dim, self.output_size)
        self.cross_entropy = nn.CrossEntropyLoss(size_average=False)

        self.adaptive = len(cutoff) > 1
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):

            if cutoff[i+1] <= cutoff[i]:
                raise ConfigurationError(f"cutoff values have to be increasing,"
                        f" while cutoff[{i+1}]({cutoff[i+1]}) > cutoff[{i}]({cutoff[i]})")

            seq = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4 ** i, False),
                nn.Linear(input_dim // 4 ** i, cutoff[i + 1] - cutoff[i], False)
            )
            self.tail.append(seq)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AdaptiveSoftmax':

        input_dim = params.pop("input_dim")
        cutoff = params.pop("cutoff")

        label_namespace = params.pop("label_namespace")
        vocab_size = vocab.get_vocab_size(label_namespace)


        if len(cutoff) > 0 and cutoff[-1] > vocab_size:
            raise ConfigurationError(f"the largest cutoff value ({cutoff[-1]}) have to be smaller"
                    f" than vocabulary size ({vocab_size})")

        logger.info("vocabulary size: %d", vocab_size)

        cutoff.append(vocab_size)

        return cls(input_dim=input_dim, vocab=vocab, cutoff=cutoff)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return 1

    def log_prob(self, w_in, device):
        """
        Parameters
        ----------
        w_in : ``Tensor``, required.
            The input of the softmax.
        device : ``required``
            The device where the calculation is conducted on.

        Returns
        -------
        prob : ``Tensor``
            The predicted future words distribution
        """
        lsm = nn.LogSoftmax(dim=1).to(device)

        head_out = self.head(w_in)

        batch_size = head_out.size(0)
        prob = torch.zeros(batch_size, self.cutoff[-1]).to(device)

        lsm_head = lsm(head_out) 
        prob.narrow(1, 0, self.output_size).add_(lsm_head.narrow(1, 0, self.output_size).data)

        for i in range(len(self.tail)):
            pos = self.cutoff[i]
            i_size = self.cutoff[i + 1] - pos
            buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1)
            buffer = buffer.expand(batch_size, i_size)
            lsm_tail = lsm(self.tail[i](w_in)) 
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return prob

    @overrides
    def forward(self, w_in, target):
        """
        Parameters
        ----------
        w_in : ``Tensor``, required
            The input of the adaptive softmax, 
        target : ``LongTensor``, required

        Returns
        -------
        loss : ``Tensor``
            A scalar loss to be optimised.

        """
        target = target.contiguous().view(-1)
        w_in = w_in.contiguous().view(-1, self.input_dim)

        batch_size = w_in.size(0)
        output = 0.0

        first_target = target.clone()

        for i in range(len(self.tail)):
            
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.data.sum() > 0:

                first_target[mask] = self.cutoff[0] + i
                second_target = target[mask].add(-self.cutoff[i])

                second_input = w_in.index_select(0, mask.nonzero().squeeze())

                second_output = self.tail[i](second_input)
                output += self.cross_entropy(second_output, second_target)

        output += self.cross_entropy(self.head(w_in), first_target)
        output /= batch_size
        return output

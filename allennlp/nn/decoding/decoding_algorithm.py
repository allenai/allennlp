from typing import Callable, Dict, List, Tuple

import torch
from torch.autograd import Variable

from allennlp.common import Params, Registrable
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.nn.decoding.decode_step import DecodeStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn import util


class DecodingAlgorithm(Registrable):
    """
    For state-based models that at every time step take one of several possible actions and then
    update their state, a ``DecodingAlgorithm`` does two things: (1) it determines how to traverse
    the exponential state space, and (2) defines a loss function over the result of that traversal.

    Concrete implementations of this abstract base class could do things like maximum marginal
    likelihood, SEARN, LaSO, or other structured learning algorithms.

    STILL IN PROGRESS.  Right now this is just doing simple maximum likelihood, in a hacky way
    (because you don't need anything this complex to do maximum likelihood).

    Parameters
    ----------
    vocab : ``Vocabulary``
        We need this so that we know the index of the start and end symbols for decoding.
    vocab_namespace : ``str``
        This tells us what namespace to look in to find the index of the start and end symbols.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 vocab_namespace: str) -> None:
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = vocab.get_token_index(START_SYMBOL, vocab_namespace)
        self._end_index = vocab.get_token_index(END_SYMBOL, vocab_namespace)

    def decode(self,
               num_steps: int,
               initial_state: DecoderState,
               decode_step: DecodeStep,
               targets: torch.Tensor = None,
               target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        decoder_state = initial_state
        finished_states = []
        for timestep in range(num_steps):
            decoder_input = self._get_decoder_input(targets,
                                                    decoder_state.outputs_so_far,
                                                    timestep,
                                                    training)
            action_log_probs, valid_actions, hidden_state = decode_step(decoder_state, decoder_input)
            if valid_actions is None:
                decoder_state = decoder_state.update(action_log_probs, hidden_state)
            if timestep == num_steps - 1:
                finished_states.append(decoder_state)

        final_state = finished_states[0]
        # (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(final_state.log_probs, 1)
        class_probabilities = logits.exp()
        predictions = torch.cat(final_state.outputs_so_far, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": predictions}
        if targets is not None:
            output_dict["loss"] = self._get_loss(logits, targets, target_mask)
            # TODO: Define metrics
        return output_dict

    def _get_decoder_input(self, targets, outputs_so_far, timestep, training):
        if training and all(torch.rand(1) >= self._scheduled_sampling_ratio):
            return targets[:, timestep]
        else:
            if timestep == 0:
                # For the first timestep, when we do not have targets, we input start symbols.
                # (batch_size,)
                return Variable(targets[:, 0].data.clone().fill_(self._start_index))
            else:
                return outputs_so_far[-1]

    @staticmethod
    def _get_loss(logits: torch.Tensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @classmethod
    def from_params(cls,
                    vocab: Vocabulary,
                    vocab_namespace: str,
                    params: Params) -> 'DecodingAlgorithm':
        choice = params.pop_choice('type', cls.list_available())
        model = cls.by_name(choice).from_params(vocab, vocab_namespace, params)
        scheduled_sampling_ratio = params.pop('scheduled_sampling_ratio', 0.0)
        return DecodingAlgorithm(vocab, vocab_namespace, scheduled_sampling_ratio)

from typing import Dict, List, Optional, Tuple

import numpy
from overrides import overrides

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.rnn import GRUCell
from torch.nn.modules.linear import Linear
from torch import nn
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import UnigramRecall


@Model.register("event2mind")
class Event2Mind(Model):
    """
    This ``Event2Mind`` class is a :class:`Model` which takes an event
    sequence, encodes it, and then uses the encoded representation to decode
    several mental state sequences.

    It is based on `the paper by Rashkin et al.
    <https://www.semanticscholar.org/paper/Event2Mind/b89f8a9b2192a8f2018eead6b135ed30a1f2144d>`_

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences.
    embedding_dropout: float, required
        The amount of dropout to apply after the source tokens have been embedded.
    encoder : ``Seq2VecEncoder``, required
        The encoder of the "encoder/decoder" model.
    max_decoding_steps : int, required
        Length of decoded sequences.
    beam_size : int, optional (default = 10)
        The width of the beam search.
    target_names: ``List[str]``, optional, (default = ['xintent', 'xreact', 'oreact'])
        Names of the target fields matching those in the ``Instance`` objects.
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 encoder: Seq2VecEncoder,
                 max_decoding_steps: int,
                 beam_size: int = 10,
                 target_names: List[str] = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None) -> None:
        super().__init__(vocab)
        target_names = target_names or ["xintent", "xreact", "oreact"]

        # Note: The original tweaks the embeddings for "personx" to be the mean
        # across the embeddings for "he", "she", "him" and "her". Similarly for
        # "personx's" and so forth. We could consider that here as a well.
        self._source_embedder = source_embedder
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        # Warning: The different decoders share a vocabulary! This may be
        # counterintuitive, but consider the case of xreact and oreact. A
        # reaction of "happy" could easily apply to both the subject of the
        # event and others. This could become less appropriate as more decoders
        # are added.
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()

        self._states = ModuleDict()
        for name in target_names:
            self._states[name] = StateDecoder(
                    num_classes,
                    target_embedding_dim,
                    self._decoder_output_dim
            )

        self._beam_search = BeamSearch(
                self._end_index,
                beam_size=beam_size,
                max_steps=max_decoding_steps
        )

    def _update_recall(self,
                       all_top_k_predictions: torch.Tensor,
                       target_tokens: Dict[str, torch.LongTensor],
                       target_recall: UnigramRecall) -> None:
        targets = target_tokens["tokens"]
        target_mask = get_text_field_mask(target_tokens)
        # See comment in _get_loss.
        # TODO(brendanr): Do we need contiguous here?
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        target_recall(
                all_top_k_predictions,
                relevant_targets,
                relevant_mask,
                self._end_index
        )

    def _get_num_decoding_steps(self,
                                target_tokens: Optional[Dict[str, torch.LongTensor]]) -> int:
        if target_tokens:
            targets = target_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end
            # symbol.  Either way, we don't have to process it. (To be clear,
            # we do still output and compare against the end symbol, but there
            # is no need to take the end symbol as input to the decoder.)
            return target_sequence_length - 1
        else:
            return self._max_decoding_steps

    @overrides
    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                **target_tokens: Dict[str, Dict[str, torch.LongTensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the target sequences.

        Parameters
        ----------
        source : ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()`` applied on the source
            ``TextField``. This will be passed through a ``TextFieldEmbedder``
            and then through an encoder.
        target_tokens : ``Dict[str, Dict[str, torch.LongTensor]]``:
            Dictionary from name to output of ``Textfield.as_array()`` applied
            on target ``TextField``. We assume that the target tokens are also
            represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, embedding_dim)
        embedded_input = self._embedding_dropout(self._source_embedder(source))
        source_mask = get_text_field_mask(source)
        # (batch_size, encoder_output_dim)
        final_encoder_output = self._encoder(embedded_input, source_mask)
        output_dict = {}

        # Perform greedy search so we can get the loss.
        if target_tokens:
            if target_tokens.keys() != self._states.keys():
                target_only = target_tokens.keys() - self._states.keys()
                states_only = self._states.keys() - target_tokens.keys()
                raise Exception("Mismatch between target_tokens and self._states. Keys in " +
                                f"targets only: {target_only} Keys in states only: {states_only}")
            total_loss = 0
            for name, state in self._states.items():
                loss = self.greedy_search(
                        final_encoder_output=final_encoder_output,
                        target_tokens=target_tokens[name],
                        target_embedder=state.embedder,
                        decoder_cell=state.decoder_cell,
                        output_projection_layer=state.output_projection_layer
                )
                total_loss += loss
                output_dict[f"{name}_loss"] = loss

            # Use mean loss (instead of the sum of the losses) to be comparable to the paper.
            output_dict["loss"] = total_loss / len(self._states)

        # Perform beam search to obtain the predictions.
        if not self.training:
            batch_size = final_encoder_output.size()[0]
            for name, state in self._states.items():
                start_predictions = final_encoder_output.new_full(
                        (batch_size,), fill_value=self._start_index, dtype=torch.long)
                start_state = {"decoder_hidden": final_encoder_output}

                # (batch_size, 10, num_decoding_steps)
                all_top_k_predictions, log_probabilities = self._beam_search.search(
                        start_predictions, start_state, state.take_step)

                if target_tokens:
                    self._update_recall(all_top_k_predictions, target_tokens[name], state.recall)
                output_dict[f"{name}_top_k_predictions"] = all_top_k_predictions
                output_dict[f"{name}_top_k_log_probabilities"] = log_probabilities

        return output_dict

    def greedy_search(self,
                      final_encoder_output: torch.LongTensor,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_embedder: Embedding,
                      decoder_cell: GRUCell,
                      output_projection_layer: Linear) -> torch.FloatTensor:
        """
        Greedily produces a sequence using the provided ``decoder_cell``.
        Returns the cross entropy between this sequence and ``target_tokens``.

        Parameters
        ----------
        final_encoder_output : ``torch.LongTensor``, required
            Vector produced by ``self._encoder``.
        target_tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()`` applied on some target ``TextField``.
        target_embedder : ``Embedding``, required
            Used to embed the target tokens.
        decoder_cell: ``GRUCell``, required
            The recurrent cell used at each time step.
        output_projection_layer: ``Linear``, required
            Linear layer mapping to the desired number of classes.
        """
        num_decoding_steps = self._get_num_decoding_steps(target_tokens)
        targets = target_tokens["tokens"]
        decoder_hidden = final_encoder_output
        step_logits = []
        for timestep in range(num_decoding_steps):
            # See https://github.com/allenai/allennlp/issues/1134.
            input_choices = targets[:, timestep]
            decoder_input = target_embedder(input_choices)
            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size, num_classes)
            output_projections = output_projection_layer(decoder_hidden)
            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))
        # (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        target_mask = get_text_field_mask(target_tokens)
        return self._get_loss(logits, targets, target_mask)

    def greedy_predict(self,
                       final_encoder_output: torch.LongTensor,
                       target_embedder: Embedding,
                       decoder_cell: GRUCell,
                       output_projection_layer: Linear) -> torch.Tensor:
        """
        Greedily produces a sequence using the provided ``decoder_cell``.
        Returns the predicted sequence.

        Parameters
        ----------
        final_encoder_output : ``torch.LongTensor``, required
            Vector produced by ``self._encoder``.
        target_embedder : ``Embedding``, required
            Used to embed the target tokens.
        decoder_cell: ``GRUCell``, required
            The recurrent cell used at each time step.
        output_projection_layer: ``Linear``, required
            Linear layer mapping to the desired number of classes.
        """
        num_decoding_steps = self._max_decoding_steps
        decoder_hidden = final_encoder_output
        batch_size = final_encoder_output.size()[0]
        predictions = [final_encoder_output.new_full(
                (batch_size,), fill_value=self._start_index, dtype=torch.long
        )]
        for _ in range(num_decoding_steps):
            input_choices = predictions[-1]
            decoder_input = target_embedder(input_choices)
            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size, num_classes)
            output_projections = output_projection_layer(decoder_hidden)
            class_probabilities = F.softmax(output_projections, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, 1)
            predictions.append(predicted_classes)
        all_predictions = torch.cat([ps.unsqueeze(1) for ps in predictions], 1)
        # Drop start symbol and return.
        return all_predictions[:, 1:]

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.FloatTensor:
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
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    def decode_all(self, predicted_indices: torch.Tensor) -> List[List[str]]:
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, List[List[str]]]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds fields for the tokens to the ``output_dict``.
        """
        for name in self._states:
            top_k_predicted_indices = output_dict[f"{name}_top_k_predictions"][0]
            output_dict[f"{name}_top_k_predicted_tokens"] = [self.decode_all(top_k_predicted_indices)]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        # Recall@10 needs beam search which doesn't happen during training.
        if not self.training:
            for name, state in self._states.items():
                all_metrics[name] = state.recall.get_metric(reset=reset)
        return all_metrics


class StateDecoder(Module):
    # pylint: disable=abstract-method
    """
    Simple struct-like class for internal use.
    """
    def __init__(self,
                 num_classes: int,
                 input_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.embedder = Embedding(num_classes, input_dim)
        self.decoder_cell = GRUCell(input_dim, output_dim)
        self.output_projection_layer = Linear(output_dim, num_classes)
        self.recall = UnigramRecall()

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        decoder_hidden = state["decoder_hidden"]
        decoder_input = self.embedder(last_predictions)
        decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
        state["decoder_hidden"] = decoder_hidden
        output_projections = self.output_projection_layer(decoder_hidden)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        return class_log_probabilities, state

from typing import Dict, List, Tuple, Union, Optional

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.softmax import Softmax
from allennlp.nn.util import get_text_field_mask

@Model.register('bidirectional-language-model')
class BidirectionalLanguageModel(Model):
    """
    This is a placeholder to figure out what other parts need to be built.
    It's not done, please do not use it yet.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 softmax: Softmax,
                 layer_norm: Optional[MaskedLayerNorm] = None,
                 dropout: float = None,
                 loss_scale_fac: Union[float, str] = 1.0,
                 remove_bos_eos: bool = True,
                 return_all_layers: bool = False) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._layer_norm = layer_norm
        self._contextualizer = contextualizer
        self._forward_dim = contextualizer.get_output_dim() // 2

        self._softmax = softmax
        self.register_buffer('_last_average_loss', torch.zeros(1))

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._loss_scale_fac = loss_scale_fac
        self._remove_bos_eos = remove_bos_eos
        self._return_all_layers = return_all_layers

    def _get_target_token_embedding(self,
                                    token_embeddings: torch.Tensor,
                                    mask: torch.Tensor,
                                    direction: int) -> torch.Tensor:
        # Need to shift the mask in the correct direction
        zero_col = token_embeddings.new_zeros(mask.size(0), 1).byte()
        if direction == 0:
            # forward direction, get token to right
            shifted_mask = torch.cat([zero_col, mask[:, 0:-1]], dim=1)
        else:
            shifted_mask = torch.cat([mask[:, 1:], zero_col], dim=1)
        return token_embeddings.masked_select(shifted_mask.unsqueeze(-1)).view(-1, self._forward_dim)

    def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      token_embeddings: torch.Tensor,
                      forward_targets: torch.Tensor,
                      backward_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lm_embeddings is shape (batch_size, timesteps, dim * 2)
        # forward_targets, backward_targets are shape (batch_size, timesteps)
        # masked with 0
        forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
        losses: List[torch.Tensor] = []
        for idx, embedding, targets in ((0, forward_embeddings, forward_targets),
                                        (1, backward_embeddings, backward_targets)):
            mask = targets > 0
            # we need to subtract 1 to undo the padding id since the softmax
            # does not include a padding dimension
            non_masked_targets = targets.masked_select(mask) - 1
            non_masked_embedding = embedding.masked_select(
                    mask.unsqueeze(-1)
            ).view(-1, self._forward_dim)
            # note: need to return average loss across forward and backward
            # directions, but total sum loss across all batches.
            # Assuming batches include full sentences, forward and backward
            # directions have the same number of samples, so sum up loss
            # here then divide by 2 just below
            if not self._softmax.tie_embeddings or not self._use_character_inputs:
                losses.append(self._softmax(non_masked_embedding, non_masked_targets))
            else:
                # we also need the token embeddings corresponding to the
                # the targets
                non_masked_token_embedding = self._get_target_token_embedding(token_embeddings, mask, idx)
                losses.append(self._softmax(non_masked_embedding,
                                            non_masked_targets,
                                            non_masked_token_embedding))

        return losses[0], losses[1]

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the averaged forward and backward LM loss from the batch.

        Tensors are sized (batch_size, timesteps) for *_targets,
        Special <S> and </S> tokens have already been added to the input
        tensors, if necessary.

        Returns:
            {'loss': averaged forward/backward negative log likelihood,
             'forward_loss': forward loss,
             'backward_loss': backward_loss,
             'lm_embeddings': (batch_size, timesteps, embed_dim) with the top
                layer contextual representations,
             'n_samples': number of non-masked target tokens in the batch
            }
        """
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(tokens)

        # We must have token_ids so that we can compute targets
        token_ids = tokens.get("tokens")
        if token_ids is None:
            raise ConfigurationError("Your data must have a 'tokens': SingleIdTokenIndexer() "
                                     "in order to use the BidirectionalLM")

        # Use token_ids to compute targets
        forward_targets = torch.zeros_like(token_ids)
        backward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]
        backward_targets[:, 1:] = token_ids[:, 0:-1]

        embeddings = self._text_field_embedder(tokens)

        if self._layer_norm is not None:
            embeddings = self._layer_norm(embeddings)

        contextual_embeddings = self._contextualizer(embeddings, mask)

        # add dropout
        if self._dropout:
            contextual_embeddings = self._dropout(contextual_embeddings)

        # compute softmax loss
        forward_loss, backward_loss = self._compute_loss(contextual_embeddings,
                                                         embeddings,
                                                         forward_targets,
                                                         backward_targets)

        num_targets = torch.sum((forward_targets > 0).long())
        if num_targets > 0:
            average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
        else:
            average_loss = torch.tensor(0.0).to(forward_targets.device)  # pylint: disable=not-callable
        # this is stored to compute perplexity if needed
        self._last_average_loss[0] = average_loss.detach().item()

        if num_targets > 0:
            # loss is directly minimized
            if self._loss_scale_fac == 'n_samples':
                scale_fac = num_targets.float()
            else:
                scale_fac = self._loss_scale_fac

            return_dict = {
                    'loss': average_loss * scale_fac,
                    'forward_loss': forward_loss * scale_fac / num_targets.float(),
                    'backward_loss': backward_loss * scale_fac / num_targets.float()
            }
        else:
            # average_loss zero tensor, return it for all
            return_dict = {
                    'loss': average_loss,
                    'forward_loss': average_loss,
                    'backward_loss': average_loss
            }

        # TODO: add non-loss functions to the output
        if self._remove_bos_eos:
            contextual_embeddings = contextual_embeddings[:, 1:-1]

        return_dict['lm_embeddings'] = contextual_embeddings

        return return_dict

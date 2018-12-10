from typing import Dict, List, Tuple, Union, Optional

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, remove_sentence_boundaries


class _SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """
    def __init__(self,
                 num_words: int,
                 embedding_dim: int) -> None:
        super().__init__()

        # TODO(joelgrus): implement tie_embeddings (maybe)
        self.tie_embeddings = False

        self.softmax_w = torch.nn.Parameter(
                torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(
                torch.matmul(embeddings, self.softmax_w) + self.softmax_b,
                dim=-1
        )

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")


@Model.register('bidirectional-language-model')
class BidirectionalLanguageModel(Model):
    """
    The ``BidirectionalLanguageModel`` applies a bidirectional "contextualizing"
    ``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
    module (defined above) to compute the language modeling loss.

    It is IMPORTANT that your bidirectional ``Seq2SeqEncoder`` does not do any
    "peeking ahead". That is, for its forward direction it should only consider
    embeddings at previous timesteps, and for its backward direction only embeddings
    at subsequent timesteps. If this condition is not met, your language model is
    cheating.

    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the indexed tokens we get in ``forward``.
    contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.
    layer_norm: ``MaskedLayerNorm``, optional (default: None)
        If provided, is applied to the noncontextualized embeddings
        before they're fed to the contextualizer.
    dropout: ``float``, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings.
    loss_scale: ``Union[float, str]``, optional (default: 1.0)
        This scaling factor is applied to the average language model loss.
        You can also specify ``"n_samples"`` in which case we compute total
        loss across all predictions.
    remove_bos_eos: ``bool``, optional (default: True)
        Typically the provided token indexes will be augmented with
        begin-sentence and end-sentence tokens. If this flag is True
        the corresponding embeddings will be removed from the return values.
    num_samples: ``int``, optional (default: None)
        If provided, the model will use ``SampledSoftmaxLoss``
        with the specified number of samples. Otherwise, it will use
        the full ``_SoftmaxLoss`` defined above.
    sparse_embeddings: ``bool``, optional (default: False)
        Passed on to ``SampledSoftmaxLoss`` if True.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 layer_norm: Optional[MaskedLayerNorm] = None,
                 dropout: float = None,
                 loss_scale: Union[float, str] = 1.0,
                 remove_bos_eos: bool = True,
                 num_samples: int = None,
                 sparse_embeddings: bool = False) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._layer_norm = layer_norm or (lambda x: x)

        if not contextualizer.is_bidirectional():
            raise ConfigurationError("contextualizer must be bidirectional")

        self._contextualizer = contextualizer
        # The dimension for making predictions just in the forward
        # (or backward) direction.
        self._forward_dim = contextualizer.get_output_dim() // 2

        # TODO(joelgrus): more sampled softmax configuration options, as needed.
        if num_samples is not None:
            self._softmax_loss = SampledSoftmaxLoss(num_words=vocab.get_vocab_size(),
                                                    embedding_dim=self._forward_dim,
                                                    num_samples=num_samples,
                                                    sparse=sparse_embeddings)
        else:
            self._softmax_loss = _SoftmaxLoss(num_words=vocab.get_vocab_size(),
                                              embedding_dim=self._forward_dim)

        self.register_buffer('_last_average_loss', torch.zeros(1))

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._loss_scale = loss_scale
        self._remove_bos_eos = remove_bos_eos

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
            if not self._softmax_loss.tie_embeddings or not self._use_character_inputs:
                losses.append(self._softmax_loss(non_masked_embedding, non_masked_targets))
            else:
                # we also need the token embeddings corresponding to the
                # the targets
                raise NotImplementedError("This requires SampledSoftmaxLoss, which isn't implemented yet.")
                # pylint: disable=unreachable
                non_masked_token_embedding = self._get_target_token_embedding(token_embeddings, mask, idx)
                losses.append(self._softmax(non_masked_embedding,
                                            non_masked_targets,
                                            non_masked_token_embedding))

        return losses[0], losses[1]

    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the averaged forward and backward LM loss from the batch.

        By convention, the input dict is required to have at least a ``"tokens"``
        entry that's the output of a ``SingleIdTokenIndexer``, which is used
        to compute the language model targets.

        If the model was instantatiated with ``remove_bos_eos=True``,
        then it is expected that each of the input sentences was augmented with
        begin-sentence and end-sentence tokens.

        Parameters
        ----------
        tokens: ``torch.Tensor``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences.

        Returns
        -------
        Dict with keys:

        ``'loss'``: ``torch.Tensor``
            averaged forward/backward negative log likelihood
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor``
            backward direction negative log likelihood
        ``'lm_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings
        """
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(source)

        # We must have token_ids so that we can compute targets
        token_ids = source.get("tokens")
        if token_ids is None:
            raise ConfigurationError("Your data must have a 'tokens': SingleIdTokenIndexer() "
                                     "in order to use the BidirectionalLM")

        # Use token_ids to compute targets
        forward_targets = torch.zeros_like(token_ids)
        backward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]
        backward_targets[:, 1:] = token_ids[:, 0:-1]

        embeddings = self._text_field_embedder(source)

        # Apply LayerNorm if appropriate.
        embeddings = self._layer_norm(embeddings)

        contextual_embeddings = self._contextualizer(embeddings, mask)

        # add dropout
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
            if self._loss_scale == 'n_samples':
                scale_factor = num_targets.float()
            else:
                scale_factor = self._loss_scale

            return_dict = {
                    'loss': average_loss * scale_factor,
                    'forward_loss': forward_loss * scale_factor / num_targets.float(),
                    'backward_loss': backward_loss * scale_factor / num_targets.float()
            }
        else:
            # average_loss zero tensor, return it for all
            return_dict = {
                    'loss': average_loss,
                    'forward_loss': average_loss,
                    'backward_loss': average_loss
            }

        if self._remove_bos_eos:
            contextual_embeddings, mask = remove_sentence_boundaries(contextual_embeddings, mask)

        return_dict.update({
                'lm_embeddings': contextual_embeddings,
                'mask': mask
        })

        return return_dict

from typing import Dict, List, Tuple, Union
import warnings

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator


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


@Model.register('language_model')
class LanguageModel(Model):
    """
    The ``LanguageModel`` applies a "contextualizing"
    ``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
    module (defined above) to compute the language modeling loss.

    If bidirectional is True,  the language model is trained to predict the next and
    previous tokens for each token in the input. In this case, the contextualizer must
    be bidirectional. If bidirectional is False, the language model is trained to only
    predict the next token for each token in the input; the contextualizer should also
    be unidirectional.

    If your language model is bidirectional, it is IMPORTANT that your bidirectional
    ``Seq2SeqEncoder`` contextualizer does not do any "peeking ahead". That is, for its
    forward direction it should only consider embeddings at previous timesteps, and for
    its backward direction only embeddings at subsequent timesteps. Similarly, if your
    language model is unidirectional, the unidirectional contextualizer should only
    consider embeddings at previous timesteps. If this condition is not met, your
    language model is cheating.

    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the indexed tokens we get in ``forward``.
    contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.

        .. deprecated:: 0.8.2
           ``contextualizer`` was deprecated in version 0.8.2 . It was
           replaced with two more flexible arguments: ``forward_contextualizer``
           and ``backward_contextualizer``, in order to enable bidirectional
           language modeling of contiguous text. It will be removed in version 0.10 .

    forward_contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings for a forward-direction LM.
        As described above, this encoder must not cheat by peeking ahead.
    backward_contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings for a backward-direction LM.
        The contextualizer should operate from left to right; the the order of the
        text in the backward inputs is assumed to have been flipped (e.g., by your
        DatasetReader). If provided, the size of its output must match that of
        the ``forward_contextualizer``.
        As described above, this encoder must not cheat by peeking ahead.
    dropout: ``float``, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    num_samples: ``int``, optional (default: None)
        If provided, the model will use ``SampledSoftmaxLoss``
        with the specified number of samples. Otherwise, it will use
        the full ``_SoftmaxLoss`` defined above.
    sparse_embeddings: ``bool``, optional (default: False)
        Passed on to ``SampledSoftmaxLoss`` if True.
    bidirectional: ``bool``, optional (default: False)
        Train a bidirectional language model, where the contextualizer
        is used to predict the next and previous token for each input token.
        This must match the bidirectionality of the contextualizer.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 forward_contextualizer: Seq2SeqEncoder = None,
                 backward_contextualizer: Seq2SeqEncoder = None,
                 dropout: float = None,
                 num_samples: int = None,
                 sparse_embeddings: bool = False,
                 bidirectional: bool = False,
                 initializer: InitializerApplicator = None) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

        # Only true when contextualizer is non-None and bidirectional is True
        self._use_contextualizer_arg = False
        if contextualizer is not None and (forward_contextualizer is not None or
                                           backward_contextualizer is not None):
            raise ConfigurationError(
                    "Cannot provide both contextualizer and either "
                    "forward_contextualizer or backward_contextualizer.")

        if contextualizer is not None:
            warnings.warn("``contextualizer`` was deprecated in version 0.8.2 . It was "
                          "replaced with two more flexible arguments: "
                          "``forward_contextualizer`` and ``backward_contextualizer``. "
                          "It will be removed in version 0.10 .",
                          DeprecationWarning)
            if contextualizer.is_bidirectional() is not bidirectional:
                raise ConfigurationError(
                        "Bidirectionality of contextualizer must match bidirectionality "
                        "of language model. "
                        f"Contextualizer bidirectional: {contextualizer.is_bidirectional()}, "
                        f"language model bidirectional: {bidirectional}")
            if contextualizer.is_bidirectional():
                # TODO (nfliu): Emit warning about cheating
                self._use_contextualizer_arg = True
            else:
                # Unidirectional LM with unidirectional contextualizer, so just set
                # forward_contextualizer to contextualizer.
                forward_contextualizer = contextualizer
                contextualizer = None
            # If self._use_contextualizer_arg is True, this is non-None. Else, it is None.
            self._contextualizer = contextualizer

        # ``contextualizer`` logic handled, do error checking for
        # forward_contextualizer and backward_contextualizer
        if bidirectional and (bool(forward_contextualizer is None) or
                              bool(backward_contextualizer is None)):
            # If we're using the contextualizer argument,
            # both forward_contextualizer and backward_contextualizer are None.
            if not self._use_contextualizer_arg:
                raise ConfigurationError(
                        "LanguageModel bidirectional is True, but did not "
                        "provide forward_contextualizer and backward_contextualizer. "
                        f"Got forward_contextualizer: {forward_contextualizer} and "
                        f"backward_contextualizer: {backward_contextualizer}")
        if not self._use_contextualizer_arg and forward_contextualizer is None:
            raise ConfigurationError(
                    "The forward_contextualizer argument is required.")
        if not bidirectional and backward_contextualizer is not None:
            raise ConfigurationError(
                    "LanguageModel bidirectional is False, so "
                    "backward_contextualizer should not be provided."
                    f"Got backward_contextualizer: {backward_contextualizer}")
        # Ensure that forward_contextualizer and backward_contextualizer
        # are unidirectional
        if forward_contextualizer and forward_contextualizer.is_bidirectional():
            raise ConfigurationError("forward_contextualizer should not be "
                                     "bidirectional.")
        if backward_contextualizer and backward_contextualizer.is_bidirectional():
            raise ConfigurationError("backward_contextualizer should not be "
                                     "bidirectional.")

        self._forward_contextualizer = forward_contextualizer
        self._backward_contextualizer = backward_contextualizer
        self._bidirectional = bidirectional

        # The dimension for making predictions just in the forward
        # (or backward) direction.
        # They must be the same. TODO (nfliu): relax this assumption
        if self._bidirectional:
            if self._use_contextualizer_arg:
                self._forward_dim = self._contextualizer.get_output_dim() // 2
            else:
                if (self._forward_contextualizer.get_output_dim() !=
                            self._backward_contextualizer.get_output_dim()):
                    raise ConfigurationError(
                            "forward_contextualizer and backward_contextualizer "
                            "must have the same output dimension. "
                            "forward_contextualizer output dimension is "
                            f"{self._forward_contextualizer.get_output_dim()}, while"
                            "backward_contextualizer output dimension is "
                            f"{self._forward_contextualizer.get_output_dim()}")
                self._forward_dim = self._forward_contextualizer.get_output_dim()
        else:
            # If bidirectional is False, self._use_contextualizer_arg is False.
            self._forward_dim = self._forward_contextualizer.get_output_dim()

        # TODO(joelgrus): more sampled softmax configuration options, as needed.
        if num_samples is not None:
            self._softmax_loss = SampledSoftmaxLoss(num_words=vocab.get_vocab_size(),
                                                    embedding_dim=self._forward_dim,
                                                    num_samples=num_samples,
                                                    sparse=sparse_embeddings)
        else:
            self._softmax_loss = _SoftmaxLoss(num_words=vocab.get_vocab_size(),
                                              embedding_dim=self._forward_dim)

        # TODO(brendanr): Output perplexity here. e^loss
        self.register_buffer('_last_average_loss', torch.zeros(1))

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        if initializer is not None:
            initializer(self)

    def _get_target_token_embeddings(self,
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
                      backward_targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
        # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
        # forward_targets, backward_targets (None in the unidirectional case) are
        # shape (batch_size, timesteps) masked with 0
        if self._bidirectional:
            forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
            backward_loss = self._loss_helper(1, backward_embeddings, backward_targets, token_embeddings)
        else:
            forward_embeddings = lm_embeddings
            backward_loss = None

        forward_loss = self._loss_helper(0, forward_embeddings, forward_targets, token_embeddings)
        return forward_loss, backward_loss

    def _loss_helper(self,  # pylint: disable=inconsistent-return-statements
                     direction: int,
                     direction_embeddings: torch.Tensor,
                     direction_targets: torch.Tensor,
                     token_embeddings: torch.Tensor) -> Tuple[int, int]:
        mask = direction_targets > 0
        # we need to subtract 1 to undo the padding id since the softmax
        # does not include a padding dimension

        # shape (batch_size * timesteps, )
        non_masked_targets = direction_targets.masked_select(mask) - 1

        # shape (batch_size * timesteps, embedding_dim)
        non_masked_embeddings = direction_embeddings.masked_select(
                mask.unsqueeze(-1)
        ).view(-1, self._forward_dim)
        # note: need to return average loss across forward and backward
        # directions, but total sum loss across all batches.
        # Assuming batches include full sentences, forward and backward
        # directions have the same number of samples, so sum up loss
        # here then divide by 2 just below
        if not self._softmax_loss.tie_embeddings or not self._use_character_inputs:
            return self._softmax_loss(non_masked_embeddings, non_masked_targets)
        else:
            # we also need the token embeddings corresponding to the
            # the targets
            raise NotImplementedError("This requires SampledSoftmaxLoss, which isn't implemented yet.")
            # pylint: disable=unreachable
            non_masked_token_embeddings = self._get_target_token_embeddings(token_embeddings, mask, direction)
            return self._softmax(non_masked_embeddings,
                                 non_masked_targets,
                                 non_masked_token_embeddings)

    def delete_softmax(self) -> None:
        """
        Remove the softmax weights. Useful for saving memory when calculating the loss
        is not necessary, e.g. in an embedder.
        """
        self._softmax_loss = None

    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self._contextualizer, 'num_layers'):
            return self._contextualizer.num_layers + 1
        else:
            raise NotImplementedError(f"Contextualizer of type {type(self._contextualizer)} " +
                                      "does not report how many layers it has.")

    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the averaged forward (and backward, if language model is bidirectional)
        LM loss from the batch.

        By convention, the input dict is required to have at least a ``"tokens"``
        entry that's the output of a ``SingleIdTokenIndexer``, which is used
        to compute the language model targets.

        Parameters
        ----------
        tokens: ``torch.Tensor``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences.

        Returns
        -------
        Dict with keys:

        ``'loss'``: ``torch.Tensor``
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor`` or ``None``
            backward direction negative log likelihood. If language model is not
            bidirectional, this is ``None``.
        ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        ``'noncontextual_token_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings
        """
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(source)

        # shape (batch_size, timesteps, embedding_size)
        embeddings = self._text_field_embedder(source)

        # Either the top layer or all layers.
        if self._use_contextualizer_arg:
            contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = (
                    self._contextualizer(embeddings, mask))
        else:
            contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = (
                    self._forward_contextualizer(embeddings, mask))
            if self._bidirectional:
                backward_contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = (
                        self._backward_contextualizer(embeddings, mask))
                # Concatenate the backward contextual embeddings to the
                # forward contextual embeddings
                if (isinstance(contextual_embeddings, list) and
                            isinstance(backward_contextual_embeddings, list)):
                    if len(contextual_embeddings) != len(backward_contextual_embeddings):
                        raise ValueError("Contextualizers produced outputs of different lengths")
                    for embedding_index, backward_embedding in enumerate(backward_contextual_embeddings):
                        contextual_embeddings[embedding_index] = torch.cat(
                                [contextual_embeddings[embedding_index], backward_embedding],
                                dim=-1)
                elif (isinstance(contextual_embeddings, torch.Tensor) and
                      isinstance(backward_contextual_embeddings, torch.Tensor)):
                    contextual_embeddings = torch.cat(
                            [contextual_embeddings, backward_contextual_embeddings], dim=-1)
                else:
                    # TODO: raise error here, since forward and backward contextual
                    # embeddings returned different things.
                    raise ValueError()

        return_dict = {}

        # If we have target tokens, calculate the loss.
        token_ids = source.get("tokens")
        if token_ids is not None:
            assert isinstance(contextual_embeddings, torch.Tensor)

            # Use token_ids to compute targets
            forward_targets = torch.zeros_like(token_ids)
            forward_targets[:, 0:-1] = token_ids[:, 1:]

            if self._bidirectional:
                backward_targets = torch.zeros_like(token_ids)
                backward_targets[:, 1:] = token_ids[:, 0:-1]
            else:
                backward_targets = None

            # add dropout
            contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

            # compute softmax loss
            forward_loss, backward_loss = self._compute_loss(contextual_embeddings_with_dropout,
                                                             embeddings,
                                                             forward_targets,
                                                             backward_targets)

            num_targets = torch.sum((forward_targets > 0).long())
            if num_targets > 0:
                if self._bidirectional:
                    average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
                else:
                    average_loss = forward_loss / num_targets.float()
            else:
                average_loss = torch.tensor(0.0).to(forward_targets.device)  # pylint: disable=not-callable
            # this is stored to compute perplexity if needed
            self._last_average_loss[0] = average_loss.detach().item()

            if num_targets > 0:
                return_dict.update({
                        'loss': average_loss,
                        'forward_loss': forward_loss / num_targets.float(),
                        'backward_loss': (backward_loss / num_targets.float()
                                          if backward_loss is not None else None),
                        'batch_weight': num_targets.float()
                })
            else:
                # average_loss zero tensor, return it for all
                return_dict.update({
                        'loss': average_loss,
                        'forward_loss': average_loss,
                        'backward_loss': average_loss if backward_loss is not None else None
                })

        return_dict.update({
                # Note: These embeddings do not have dropout applied.
                'lm_embeddings': contextual_embeddings,
                'noncontextual_token_embeddings': embeddings,
                'mask': mask
        })

        return return_dict

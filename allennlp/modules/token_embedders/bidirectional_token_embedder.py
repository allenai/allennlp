from typing import Dict, List, Tuple, Union, Optional

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import ScalarMix
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, BidirectionalTransformerEncoder
from allennlp.nn.util import get_text_field_mask, remove_sentence_boundaries
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation


@TokenEmbedder.register('bidirectional_token_embedder')
class BidirectionalTokenEmbedder(TokenEmbedder):
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
                 weight_file: str,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: BidirectionalTransformerEncoder,
                 layer_norm: Optional[MaskedLayerNorm] = None,
                 dropout: float = None,
                 remove_bos_eos: bool = True,
                 requires_grad: bool = False) -> None:
        super().__init__()
        self._text_field_embedder = text_field_embedder
        self._layer_norm = layer_norm or (lambda tensor, mask: tensor)

        if not contextualizer.is_bidirectional():
            raise ConfigurationError("contextualizer must be bidirectional")
        if not contextualizer._return_all_layers:
            raise ConfigurationError("contextualizer must return all layers")

        self._contextualizer = contextualizer
        # The dimension for making predictions just in the forward
        # (or backward) direction.
        self._forward_dim = contextualizer.get_output_dim() // 2

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._remove_bos_eos = remove_bos_eos
        self._scalar_mix = ScalarMix(mixture_size=contextualizer.num_layers + 1, do_layer_norm=False, trainable=True)

        state_dict = torch.load(weight_file)
        self.load_state_dict(state_dict, strict=False)

        for param in self._text_field_embedder.parameters():
            param.requires_grad = requires_grad
        for param in self._contextualizer.parameters():
            param.requires_grad = requires_grad

    def get_output_dim(self) -> int:
        return self._contextualizer.output_dim

    def forward(self,  # type: ignore
                inputs: torch.LongTensor) -> Dict[str, torch.Tensor]:
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
        source = {"token_characters": inputs}
        mask = get_text_field_mask(source)
        #print(f"BRR: {source}")

        # shape (batch_size, sentence_length + 2, embedding_size)
        embeddings = self._text_field_embedder(source)

        # Apply LayerNorm if appropriate.
        embeddings = self._layer_norm(embeddings, mask)

        contextual_embeddings = self._contextualizer(embeddings, mask)

        # To match contextualized dimension.
        double_character_embeddings = torch.cat([embeddings, embeddings], -1)
        if double_character_embeddings.size(-1) != contextual_embeddings[0].size(-1):
            raise Exception("Incorrect sizes")

        contextual_embeddings.append(double_character_embeddings)
        averaged_embeddings = self._scalar_mix(contextual_embeddings)

        # add dropout
        averaged_embeddings = self._dropout(averaged_embeddings)

        if self._remove_bos_eos:
            averaged_embeddings, mask = remove_sentence_boundaries(averaged_embeddings, mask)

        return averaged_embeddings

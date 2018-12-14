from typing import Dict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules import ScalarMix
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.seq2seq_encoders import BidirectionalLanguageModelTransformer
from allennlp.nn.util import device_mapping, get_text_field_mask, remove_sentence_boundaries


@TokenEmbedder.register('bidirectional_token_embedder')
class BidirectionalTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of representations from a bidirectional language model with a
    transformer contextualizer.

    Parameters
    ----------
    weight_file : ``str``, required
        An model weights file, e.g. best.th, from a BidirectionalLanguageModel trained using a
        BidirectionalLanguageModelTransformer as a contextualizer.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the character ids we get in ``forward``.
    contextualizer : ``BidirectionalLanguageModelTransformer``, required
        Used to "contextualize" the embeddings.
    dropout : ``float``, optional.
        The dropout value to be applied to the representations.
    remove_bos_eos: ``bool``, optional (default: True)
        Typically the provided token indexes will be augmented with
        begin-sentence and end-sentence tokens. If this flag is True
        the corresponding embeddings will be removed from the return values.
    requires_grad : ``bool``, optional (default: False)
        If True, compute gradient of bidirectional language model parameters for fine tuning.
    """
    def __init__(self,
                 weight_file: str,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: BidirectionalLanguageModelTransformer,
                 dropout: float = None,
                 remove_bos_eos: bool = True,
                 requires_grad: bool = False) -> None:
        super().__init__()
        self._text_field_embedder = text_field_embedder

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

        state_dict = torch.load(weight_file, map_location=device_mapping(-1))
        self.load_state_dict(state_dict, strict=False)

        for param in self._text_field_embedder.parameters():
            param.requires_grad = requires_grad
        for param in self._contextualizer.parameters():
            param.requires_grad = requires_grad

    def get_output_dim(self) -> int:
        return self._contextualizer.output_dim

    def forward(self,  # type: ignore
                inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

        Returns
        -------
        The bidirectional language model representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        # pylint: disable=arguments-differ
        source = {"token_characters": inputs}
        mask = get_text_field_mask(source)

        # shape (batch_size, sentence_length + 2, embedding_size)
        embeddings = self._text_field_embedder(source)

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

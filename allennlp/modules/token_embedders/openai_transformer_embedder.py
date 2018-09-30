import torch

from allennlp.modules.openai_transformer import OpenaiTransformer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import get_range_vector, get_device_of


@TokenEmbedder.register("openai_transformer_embedder")
class OpenaiTransformerEmbedder(TokenEmbedder):
    """
    Takes a byte-pair representation of a batch of sentences
    (as produced by the ``OpenaiTransformerBytePairIndexer``)
    and generates a `ScalarMix` of the corresponding contextual embeddings.



    Parameters
    ----------
    transformer: ``OpenaiTransformer``, required.
        The ``OpenaiTransformer`` module used for the embeddings.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    """
    def __init__(self,
                 transformer: OpenaiTransformer,
                 top_layer_only: bool = False) -> None:
        super().__init__()

        self._transformer = transformer
        self._top_layer_only = top_layer_only
        if not top_layer_only:
            self._scalar_mix = ScalarMix(transformer.num_output_layers, do_layer_norm=False)

    def get_output_dim(self):
        """
        The last dimension of the output, not the shape.
        """
        return self._transformer.embed.embedding_dim

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required
            A ``(batch_size, num_timesteps)`` tensor representing the byte-pair encodings
            for the current batch.
        offsets: ``torch.Tensor``, required
            A ``(batch_size, max_sequence_length)`` tensor representing the word offsets
            for the current batch.

        Returns
        -------
        ``[torch.Tensor]``
            An embedding representation of the input sequence
            having shape ``(batch_size, sequence_length, embedding_dim)``
        """
        # pylint: disable=arguments-differ
        batch_size, num_timesteps = inputs.size()

        # the transformer embedding consists of the byte pair embeddings,
        # the special embeddings and the position embeddings.
        # the position embeddings are always at least self._transformer.n_ctx,
        # but may be longer.
        # the transformer "vocab" consists of the actual vocab and the
        # positional encodings. Here we want the count of just the former.
        vocab_size = self._transformer.vocab_size - self._transformer.n_ctx

        # vocab_size, vocab_size + 1, ...
        positional_encodings = get_range_vector(num_timesteps, device=get_device_of(inputs)) + vocab_size

        # Combine the inputs with positional encodings
        batch_tensor = torch.stack([
                inputs,   # (batch_size, num_timesteps)
                positional_encodings.expand(batch_size, num_timesteps)
        ], dim=-1)

        byte_pairs_mask = inputs != 0

        # Embeddings is num_output_layers x (batch_size, num_timesteps, embedding_dim)
        layer_activations = self._transformer(batch_tensor)

        # Output of scalar_mix is (batch_size, num_timesteps, embedding_dim)
        if self._top_layer_only:
            mix = layer_activations[-1]
        else:
            mix = self._scalar_mix(layer_activations, byte_pairs_mask)

        # These embeddings are one per byte-pair, but we want one per original _word_.
        # So we choose the embedding corresponding to the last byte pair for each word,
        # which is captured by the ``offsets`` input.
        if offsets is not None:
            range_vector = get_range_vector(batch_size, device=get_device_of(mix)).unsqueeze(1)
            last_byte_pair_embeddings = mix[range_vector, offsets]
        else:
            # allow to return all byte pairs by passing no offsets
            seq_len = (byte_pairs_mask > 0).long().sum(dim=1).max()
            last_byte_pair_embeddings = mix[:, :seq_len]

        return last_byte_pair_embeddings

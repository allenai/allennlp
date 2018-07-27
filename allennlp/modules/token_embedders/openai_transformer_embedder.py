from typing import List
import torch

from allennlp.modules import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.openai_transformer import OpenaiTransformer


@TokenEmbedder.register("openai_transformer_embedder")
class OpenaiTransformerEmbedder(TokenEmbedder):
    """
    Takes a byte-pair representation of a batch of sentences
    (as produced by the ``OpenaiTransformerBytePairIndexer``)
    and generates the corresponding contextual embeddings.

    Parameters
    ----------
    transformer: ``OpenaiTransformer``, required.
        The ``OpenaiTransformer`` module used for the embeddings.
    num_output_representations: ``int``, optional (default: 1)
        How many "scalar mixes" of the embedding layers to return.
        (Currently only implemented for n = 1.)
    do_layer_norm: ``bool`` (default: ``False``)
        Whether the ``ScalarMix``es should use layer norm.
    """
    def __init__(self,
                 transformer: OpenaiTransformer,
                 num_output_representations: int = 1,
                 do_layer_norm: bool = False) -> None:
        super().__init__()

        if num_output_representations > 1:
            raise NotImplementedError("more than 1 output representation is not implemented")

        self._transformer = transformer
        self._num_output_representations = num_output_representations
        self._do_layer_norm = do_layer_norm

        self._scalar_mixes: List[ScalarMix] = []

        for k in range(num_output_representations):
            scalar_mix = ScalarMix(transformer.num_output_layers, do_layer_norm=do_layer_norm)
            self.add_module(f'scalar_mix_{k}', scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self):
        """
        The last dimension of the output, not the shape.
        """
        return self._transformer.embed.embedding_dim

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
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

        # the transformer "vocab" consists of the actual vocab and the
        # positional encodings. Here we want the count of just the former.
        vocab_size = self._transformer.vocab_size - self._transformer.n_ctx

        # Combine the inputs with positional encodings
        batch_tensor = torch.zeros((batch_size, num_timesteps, 2), dtype=torch.long)
        batch_tensor[:, :, 0] = inputs
        batch_tensor[:, :, 1] = torch.arange(vocab_size, vocab_size + num_timesteps)

        byte_pairs_mask = inputs != 0

        # Embeddings is num_output_layers x (batch_size, num_timesteps, embedding_dim)
        layer_activations = self._transformer(batch_tensor)

        # Output of each scalar_mix is (batch_size, num_timesteps, embedding_dim)
        mixes = [scalar_mix(layer_activations, byte_pairs_mask) for scalar_mix in self._scalar_mixes]

        # These embeddings are one per byte-pair, but we want one per original _word_.
        # Fortunately, we have the ``offsets`` which indicate the last byte-pair for
        # each original word. Here we just choose that last byte-pair, although you
        # could imagine doing something more sophisticated.
        _, max_sequence_length = offsets.size()
        _, _, embedding_dim = mixes[0].size()

        # TODO(joelgrus): vectorize?
        embeddings = [torch.zeros(batch_size, max_sequence_length, embedding_dim) for _ in mixes]
        for i in range(batch_size):
            last_offset = -1
            for j in range(max_sequence_length):
                offset = offsets[i, j]
                if offset <= last_offset:
                    break

                for embedding, mix in zip(embeddings, mixes):
                    embedding[i, j] = mix[i, offset]

                last_offset = offset

        return embeddings[0]

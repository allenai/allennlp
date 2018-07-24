from typing import List
import torch

from allennlp.modules import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.openai_transformer import OpenaiTransformer


@TokenEmbedder.register("openai_transformer_embedder")
class OpenaiTransformerEmbedder(TokenEmbedder):
    """
    Parameters
    ----------
    weights_path : ``str``, required.
        Path to the serialized OpenAI transformer model.
    """
    def __init__(self,
                 transformer_model_path: str,
                 num_output_representations: int = 1,
                 do_layer_norm: bool = False,
                 requires_grad: bool = False) -> None:
        super().__init__()

        if num_output_representations > 1:
            raise NotImplementedError("more than 1 output representation is not implemented")

        self._transformer = OpenaiTransformer(requires_grad=requires_grad)
        self._transformer.load_weights(transformer_model_path)
        self._num_output_representations = num_output_representations
        self._do_layer_norm = do_layer_norm

        self._scalar_mixes: List[ScalarMix] = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(13, do_layer_norm=do_layer_norm)
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
            A ``(batch_size, timesteps)`` tensor representing the byte-pair encodings
            for the current batch.
        offsets: ``torch.Tensor``, required
            A ``(batch_size, max_sequence_length)`` tensor representing the word offsets
            for the current batch.

        Returns
        -------
        Dict with keys
        ``'transformer_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of representations for the input sequence.
            Each is shape ``(batch_size, sentence_length, embedding_dim)``
        ``'mask'``: ``torch.Tensor``
            Shape ``(batch_size, sentence_length)`` long tensor with sequence mask.
        """
        # pylint: disable=arguments-differ
        batch_size, num_timesteps = inputs.size()
        total_vocab_size = self._transformer.vocab_size - self._transformer.n_ctx

        batch_tensor = torch.zeros((batch_size, num_timesteps, 2), dtype=torch.long)
        batch_tensor[:, :, 0] = inputs
        batch_tensor[:, :, 1] = torch.arange(total_vocab_size, total_vocab_size + num_timesteps)

        byte_pairs_mask = inputs != 0

        # Embeddings is 13 x (batch_size, timesteps, embedding_dim)
        layer_activations = self._transformer(batch_tensor)

        # Each scalar_mix is (batch_size, timesteps, embedding_dim)
        mixes = [scalar_mix(layer_activations, byte_pairs_mask) for scalar_mix in self._scalar_mixes]

        # Now take the last byte pair for each word
        _, max_sequence_length = offsets.size()
        _, _, embedding_dim = mixes[0].size()

        embeddings = [torch.zeros(batch_size, max_sequence_length, embedding_dim) for _ in mixes]
        mask = torch.zeros(batch_size, max_sequence_length)

        # TODO(joelgrus) no for loop
        for i in range(batch_size):
            last_offset = -1
            for j in range(max_sequence_length):
                offset = offsets[i, j]
                if offset <= last_offset:
                    break

                mask[i, j] = 1

                for embedding, mix in zip(embeddings, mixes):
                    embedding[i, j] = mix[i, offset]

                last_offset = offset

        return embeddings[0]

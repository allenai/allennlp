from overrides import overrides
import torch

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, TokenEmbedder


@TokenEmbedder.register("pretrained_transformer_pretokenized")
class PretrainedTransformerPretokenizedEmbedder(PretrainedTransformerEmbedder):
    """
    Use this embedder when input comes from pre-tokenized text, `PretrainedTransformerTokenizer` was
    not used in the dataset loader, and `PretrainedTransformerPretokenizedIndexer` independently
    tokenized each word into subword wordpieces.
    """

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.LongTensor,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces].
        mask: torch.LongTensor
            Shape: [batch_size, num_orig_tokens].
        offsets: torch.LongTensor
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        orig_token_mask: torch.LongTensor
            Shape: [batch_size, num_wordpieces].

        # Returns:

        Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        device = token_ids.device

        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = super().forward(token_ids, wordpiece_mask)

        batch_size, _, embedding_size = embeddings.size()
        num_orig_tokens = offsets.size(1)

        orig_embeddings = torch.FloatTensor(
            batch_size, num_orig_tokens, embedding_size
        ).to(device)
        for batch_idx in range(batch_size):  # TODO: do we have to use loops?
            for token_idx in range(num_orig_tokens):
                if not mask[batch_idx, token_idx]:
                    continue

                start_offset, end_offset = offsets[batch_idx, token_idx]
                end_offset += 1  # inclusive to exclusive
                assert wordpiece_mask[batch_idx, start_offset:end_offset].bool().all()
                embedding = embeddings[batch_idx, start_offset:end_offset].mean(0)
                orig_embeddings[batch_idx, token_idx] = embedding

        return orig_embeddings

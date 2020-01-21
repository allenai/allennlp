from overrides import overrides
import torch

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, TokenEmbedder
from allennlp.nn import util


@TokenEmbedder.register("pretrained_transformer_pretokenized")
class PretrainedTransformerPretokenizedEmbedder(PretrainedTransformerEmbedder):
    """
    Use this embedder when input comes from pre-tokenized text, `PretrainedTransformerTokenizer` was
    not used in the dataset loader, and `PretrainedTransformerPretokenizedIndexer` independently
    tokenized each word into subword wordpieces. This embedder sums the wordpiece representations
    to get word-level representations.
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
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = super().forward(token_ids, wordpiece_mask)

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings, offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / span_embeddings_len

        return orig_embeddings

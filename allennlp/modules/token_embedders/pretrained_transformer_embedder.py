import math
from typing import Optional

from overrides import overrides
from transformers.modeling_auto import AutoModel
from transformers.tokenization_auto import AutoTokenizer
import torch
import torch.nn.functional as F

from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_len : `int`, optional (default = -1)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_len` option on the
        `PretrainedTransformerIndexer`.
    """

    def __init__(self, model_name: str, max_len: int = -1) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self._max_len = max_len
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        (
            self._num_added_start_tokens,
            self._num_added_end_tokens,
        ) = PretrainedTransformerIndexer.determine_num_special_tokens_added(tokenizer)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

    @overrides
    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces if max_len <= 0 else num_segment_concat_wordpieces].
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, i.e. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: torch.LongTensor
             Shape: [batch_size, num_wordpieces].
        segment_concat_mask: torch.LongTensor, optional.
            Shape: [batch_size, num_segment_concat_wordpieces].

        # Returns:

        Shape: [batch_size, num_wordpieces, embedding_size].
        """
        if (
            type_ids is not None
            and type_ids.max()
            >= self.transformer_model.embeddings.token_type_embeddings.num_embeddings
            == 1
        ):
            raise ValueError("Found type ids too large for the chosen transformer model.")

        batch_size = token_ids.size(0)

        if self._max_len > 0:
            # [ [CLS] A B C [SEP] [CLS] D E F [SEP] ] ->
            # [ [ [CLS] A B C [SEP] ], [ [CLS] D E F [SEP] ] ]
            num_segment_concat_wordpieces = token_ids.size(1)
            num_segments = math.ceil(num_segment_concat_wordpieces / self._max_len)
            padded_length = num_segments * self._max_len
            length_to_pad = padded_length - num_segment_concat_wordpieces

            def fold(tensor):  # Shape: [batch_size, num_segment_concat_wordpieces]
                # Shape: [batch_size, num_segments * self._max_len]
                tensor = F.pad(tensor, [0, length_to_pad])
                # Shape: [batch_size * num_segments, self._max_len]
                return tensor.reshape(-1, self._max_len)

            token_ids = fold(token_ids)
            segment_concat_mask = fold(segment_concat_mask)

        transformer_mask = segment_concat_mask if self._max_len > 0 else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_len > 0: [batch_size * num_segments, self._max_len, embedding_size]
        embeddings = self.transformer_model(
            input_ids=token_ids, token_type_ids=type_ids, attention_mask=transformer_mask
        )[0]
        embedding_size = embeddings.size(2)

        if self._max_len > 0:
            # We truncate the start and end tokens for all segments, recombine the segments,
            # and manually add back the start tokens. We generally don't need to add back
            # the end tokens -- because the last segment is usually shorter than self._max_len,
            # the final end tokens usually won't be touched in our truncation process.
            #
            # There are special cases where sequences have a full last segment. The only issue
            # in this case is that the [SEP] embedding will be wrong. Nevertheless, [SEP] embeddings
            # are rarely, if ever, used, so it shouldn't be a huge problem. However, we do remedy
            # this case when such sequences with a full last segment are the longest ones in a batch
            # by adding back the last token representations.

            # We want to remove all segment-level special tokens but maintain sequence-level ones
            num_wordpieces = (
                num_segment_concat_wordpieces - (num_segments - 1) * self._num_added_tokens
            )

            embeddings = embeddings.reshape(
                batch_size, num_segments * self._max_len, embedding_size
            )
            # Shape: (batch_size, self._num_added_start_tokens, embedding_size)
            start_token_embeddings = embeddings[:, : self._num_added_start_tokens, :]
            # See comment above -- remedy when the longest sequences have full last segments.
            # This is not necessarily the end token embeddings when sequences aren't full.
            # Shape: (batch_size, self._num_added_end_tokens, embedding_size)
            last_token_embeddings = embeddings[:, -self._num_added_end_tokens :, :]

            embeddings = embeddings.reshape(batch_size, num_segments, self._max_len, embedding_size)
            embeddings = embeddings[
                :, :, self._num_added_start_tokens : -self._num_added_end_tokens, :
            ]  # truncate segment-level start/end tokens
            embeddings = embeddings.reshape(batch_size, -1, embedding_size)  # flatten
            embeddings = torch.cat([start_token_embeddings, embeddings, last_token_embeddings], 1)
            embeddings = embeddings[:, :num_wordpieces, :]

        return embeddings

from overrides import overrides
from transformers.modeling_auto import AutoModel
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.
    """

    def __init__(self, model_name: str, intra_word_tokenized: bool = False) -> None:
        """
        # Parameters

        model_name : ``str``, required.
            The name of the transformer model to use.
        intra_word_tokenized: ``bool``, optional (default = False)
            Whether or not the input comes from intra-word tokenization in the indexer. If so, we
            pool representations of wordpieces of a word to get word-level representations, and
            `offsets` must be provided in `forward()`. Should be set to the same value as the
            ``intra_word_tokenization`` option on the :class:`PretrainedTransformerIndexer`.
        """
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self._intra_word_tokenized = intra_word_tokenized
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        wordpiece_mask: torch.LongTensor = None,
        offsets: torch.LongTensor = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces].
        mask: torch.LongTensor
            Shape: [batch_size, num_orig_tokens].
        wordpiece_mask: torch.LongTensor
            Shape: [batch_size, num_wordpieces].
        offsets: torch.LongTensor
            Shape: [batch_size, num_orig_tokens, 2].
            token_ids[i][offsets[i][j][0]:offsets[i][j][1]] corresponds to the original j-th token
            from the i-th batch.

        # Returns:
        Shape: [batch_size, (num_wordpieces or num_orig_tokens), embedding_size].
        """
        if self._intra_word_tokenized:
            if wordpiece_mask is None or offsets is None:
                raise ValueError(
                    "`wordpiece_mask` and `offsets` must be set if `intra_word_tokenized == True`."
                )

        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self.transformer_model(
            input_ids=token_ids,
            attention_mask=(wordpiece_mask if self._intra_word_tokenized else mask),
        )[0]

        if self._intra_word_tokenized:
            batch_size, _, embedding_size = embeddings.size()
            num_orig_tokens = offsets.size(1)

            orig_embeddings = torch.FloatTensor(batch_size, num_orig_tokens, embedding_size)
            for batch_idx in range(batch_size):  # TODO: do we have to use loops?
                for token_idx in range(num_orig_tokens):
                    if not mask[batch_idx, token_idx]:
                        continue

                    start_offset, end_offset = offsets[batch_idx, token_idx]
                    end_offset += 1  # inclusive to exclusive
                    assert wordpiece_mask[batch_idx, start_offset:end_offset].bool().all()
                    embedding = embeddings[batch_idx, start_offset:end_offset].mean(0)
                    orig_embeddings[batch_idx, token_idx] = embedding

            embeddings = orig_embeddings

        return embeddings

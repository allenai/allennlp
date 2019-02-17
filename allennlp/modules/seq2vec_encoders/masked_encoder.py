from typing import Dict, Tuple
import torch

from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask

@Seq2VecEncoder.register("masked_encoder")
class MaskedEncoder(Seq2VecEncoder):
    """
    This ``MaskedEncoder``. This class wraps Pytorch RNN with embedding and masking

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    rnn : ``Seq2VecEncoder``, required
        The encoder of the "encoder/decoder" model
    """

    def __init__(self,
                 source_embedder: TextFieldEmbedder,
                 rnn: Seq2VecEncoder) -> None:
        super(MaskedEncoder, self).__init__()
        self._source_embedder = source_embedder
        self.rnn = rnn

    def get_input_dim(self) -> int:
        return self.rnn.get_input_dim()

    def get_output_dim(self) -> int:
        return self.rnn.get_output_dim

    def forward(self, source_tokens: Dict[str, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make a forward pass of the encoder, then returning the hidden state.
        """
        embeddings = self._source_embedder(source_tokens)
        mask = get_text_field_mask(source_tokens)
        final_state = self.rnn(embeddings, mask)

        return final_state

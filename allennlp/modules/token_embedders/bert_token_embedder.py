import logging

import torch

from allennlp.common.file_utils import cached_path
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
import allennlp.modules.token_embedders._bert_huggingface as bert
from allennlp.nn import util

logger = logging.getLogger(__name__)


class BertEmbedder(TokenEmbedder):
    """
    This class is not registered, because you probably want to use
    a specific subclass corresponding to a trained model.
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 intermediate_size: int,
                 hidden_act: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 max_position_embeddings: int,
                 type_vocab_size: int,
                 initializer_range: float,
                 init_checkpoint: str = None) -> None:
        super().__init__()

        self.output_dim = hidden_size

        config = bert.BertConfig(vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                                 intermediate_size, hidden_act, hidden_dropout_prob,
                                 attention_probs_dropout_prob, max_position_embeddings, type_vocab_size,
                                 initializer_range)

        self.bert_model = bert.BertModel(config)

        if init_checkpoint is not None:
            logger.info(f"loading pretrained BERT model from {init_checkpoint}")
            state_dict = torch.load(cached_path(init_checkpoint))
            self.bert_model.load_state_dict(state_dict)
        else:
            logger.warning("no checkpoint provided for BERT model, you're using garbage weights!")

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                input_ids: torch.LongTensor,        # ([[31, 51, 99], [15, 5, 0]])
                input_mask: torch.LongTensor,       # ([[1, 1, 1], [1, 1, 0]])
                token_type_ids: torch.LongTensor,   # ([[0, 0, 1], [0, 2, 0]])
                offsets: torch.LongTensor = None):
        # pylint: disable=arguments-differ
        all_encoder_layers, _ = self.bert_model(input_ids, input_mask, token_type_ids)
        sequence_output = all_encoder_layers[-1]

        if offsets is None:
            return sequence_output
        else:
            batch_size = input_ids.size(0)
            range_vector = util.get_range_vector(batch_size,
                                                 device=util.get_device_of(sequence_output)).unsqueeze(1)
            return sequence_output[range_vector, offsets]


@TokenEmbedder.register("bert-base-uncased")
class BertBaseUncased(BertEmbedder):
    def __init__(self, init_checkpoint: str = None) -> None:
        super().__init__(vocab_size=30522,
                         hidden_size=768,
                         num_hidden_layers=12,
                         num_attention_heads=12,
                         intermediate_size=3072,
                         hidden_act="gelu",
                         hidden_dropout_prob=0.1,
                         attention_probs_dropout_prob=0.1,
                         max_position_embeddings=512,
                         type_vocab_size=2,
                         initializer_range=0.02,
                         init_checkpoint=init_checkpoint)

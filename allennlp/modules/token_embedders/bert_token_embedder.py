import logging

import torch

from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from allennlp.common.file_utils import cached_path
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util

logger = logging.getLogger(__name__)


class BertEmbedder(TokenEmbedder):
    """
    Don't use this class, use ``PretrainedBertEmbedder`` for one of the
    named pretrained models or ``CustomBertEmbedder`` for a custom model.
    """
    def __init__(self,
                 bert_model: BertModel) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.output_dim = bert_model.config.hidden_size

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


@TokenEmbedder.register("bert-pretrained")
class PretrainedBertEmbedder(BertEmbedder):
    def __init__(self, pretrained_model_name: str) -> None:
        super().__init__(BertModel.from_pretrained(pretrained_model_name))

@TokenEmbedder.register("bert-custom")
class CustomBertEmbedder(BertEmbedder):
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
        self.output_dim = hidden_size

        config = BertConfig(vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                            intermediate_size, hidden_act, hidden_dropout_prob,
                            attention_probs_dropout_prob, max_position_embeddings, type_vocab_size,
                            initializer_range)

        bert_model = BertModel(config)

        if init_checkpoint is not None:
            logger.info(f"loading pretrained BERT model from {init_checkpoint}")
            state_dict = torch.load(cached_path(init_checkpoint))
            bert_model.load_state_dict(state_dict)
        else:
            logger.warning("no checkpoint provided for BERT model, you're using garbage weights!")

        super().__init__(bert_model)

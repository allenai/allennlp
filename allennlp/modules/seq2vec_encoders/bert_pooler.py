from typing import Union

from overrides import overrides

import torch
import torch.nn
from pytorch_pretrained_bert import BertModel

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel


@Seq2VecEncoder.register("bert_pooler")
class BertPooler(Seq2VecEncoder):
    """
    The pooling layer at the end of the BERT model. This returns an embedding for the
    [CLS] token, after passing it through a non-linear tanh activation; the non-linear layer
    is also part of the BERT model. If you want to use the pretrained BERT model
    to build a classifier and you want to use the AllenNLP token-indexer ->
    token-embedder -> seq2vec encoder setup, this is the Seq2VecEncoder to use.
    (For example, if you want to experiment with other embedding / encoding combinations.)

    If you just want to train a BERT classifier, it's simpler to just use the
    ``BertForClassification`` model.

    Parameters
    ----------
    pretrained_model : ``Union[str, BertModel]``
        The pretrained BERT model to use. If this is a string,
        we will call ``BertModel.from_pretrained(pretrained_model)``
        and use that.
    requires_grad : ``bool``, optional, (default = True)
        If True, the weights of the pooler will be updated during training.
        Otherwise they will not.
    dropout : ``float``, optional, (default = 0.0)
        Amount of dropout to apply after pooling
    """
    def __init__(self,
                 pretrained_model: Union[str, BertModel],
                 requires_grad: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()

        if isinstance(pretrained_model, str):
            model = PretrainedBertModel.load(pretrained_model)
        else:
            model = pretrained_model

        self._dropout = torch.nn.Dropout(p=dropout)

        self.pooler = model.pooler
        for param in self.pooler.parameters():
            param.requires_grad = requires_grad
        self._embedding_dim = model.config.hidden_size

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):  # pylint: disable=arguments-differ,unused-argument
        pooled = self.pooler(tokens)
        pooled = self._dropout(pooled)
        return pooled

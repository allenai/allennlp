from overrides import overrides

import torch
from pytorch_pretrained_bert import BertModel

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

@Seq2VecEncoder.register("bert_pooler")
class BertPooler(Seq2VecEncoder):
    """
    The pooling layer at the end of the BERT model. If you want to use the pretrained
    BERT model to build a classifier and you want to use the AllenNLP token-indexer ->
    token-embedder -> seq2vec encoder setup, this is the Seq2VecEncoder to use.
    (For example, if you want to experiment with other embedding / encoding combinations.)

    If you just want to train a BERT classifier, it's simpler to just use the
    ``BertForClassification`` model.

    Parameters
    ----------
    pretrained_model: ``str``
        The pretrained BERT model to use.
    """
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()

        # TODO(joelgrus): it's inefficient to load the model here and (presumably) also in the
        # BertTokenEmbedder, is there a way to load it only once?
        model = BertModel.from_pretrained(pretrained_model)
        self.pooler = model.pooler
        self._embedding_dim = model.config.hidden_size

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):  # pylint: disable=arguments-differ,unused-argument
        return self.pooler(tokens)

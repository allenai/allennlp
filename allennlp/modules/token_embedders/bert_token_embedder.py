"""
A ``TokenEmbedder`` which uses one of the BERT models
(https://github.com/google-research/bert)
to produce embeddings.

At its core it uses Hugging Face's PyTorch implementation
(https://github.com/huggingface/pytorch-pretrained-BERT),
so thanks to them!
"""
import logging

import torch

from pytorch_pretrained_bert.modeling import BertModel

from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util

logger = logging.getLogger(__name__)


class BertEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
    Should be paired with a ``BertIndexer``, which produces wordpiece ids.

    Most likely you probably want to use ``PretrainedBertEmbedder``
    for one of the named pretrained models, not this base class.

    Parameters
    ----------
    bert_model: ``BertModel``
        The BERT model being wrapped.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    """
    def __init__(self,
                 bert_model: BertModel,
                 top_layer_only: bool = False) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.output_dim = bert_model.config.hidden_size
        if not top_layer_only:
            self._scalar_mix = ScalarMix(bert_model.config.num_hidden_layers,
                                         do_layer_norm=False)
        else:
            self._scalar_mix = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                input_ids: torch.LongTensor,
                offsets: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids: ``torch.LongTensor``
            The wordpiece ids for each input sentence.
        offsets: ``torch.LongTensor``, optional (default = None)
            The BERT embeddings are one per wordpiece. However it's possible/likely
            you might want one per original token. In that case, ``offsets``
            should represent the indices of the last wordpiece for each original token.

            That is, if you had the sentence "Definitely not", and if the corresponding
            wordpieces were ["Def", "##in", "##ite", "##ly", "not"], then the input_ids
            would be 5 wordpiece ids, and the offsets would be [3, 4]. If offsets are
            provided, the returned tensor will contain the _last_ wordpiece embedding
            for each token, and (in particular) will contain one embedding per token.
            If offsets are not provided, the entire tensor of wordpiece embeddings
            will be returned.
        token_type_ids: ``torch.LongTensor``, optional (default = None)
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """
        # pylint: disable=arguments-differ
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        all_encoder_layers, _ = self.bert_model(input_ids, input_mask, token_type_ids)
        if self._scalar_mix is not None:
            mix = self._scalar_mix(all_encoder_layers, input_mask)
        else:
            mix = all_encoder_layers[-1]


        if offsets is None:
            return mix
        else:
            batch_size = input_ids.size(0)
            range_vector = util.get_range_vector(batch_size,
                                                 device=util.get_device_of(mix)).unsqueeze(1)
            return mix[range_vector, offsets]


@TokenEmbedder.register("bert-pretrained")
class PretrainedBertEmbedder(BertEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model_name: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .tar.gz file with the model weights.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L41
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of BERT parameters for fine tuning.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    """
    def __init__(self, pretrained_model: str, requires_grad: bool = False, top_layer_only: bool = False) -> None:
        model = BertModel.from_pretrained(pretrained_model)

        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(bert_model=model, top_layer_only=top_layer_only)

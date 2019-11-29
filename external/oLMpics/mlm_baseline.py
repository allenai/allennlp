# TODO: projection dropout with ELMO
#   l2 reg with ELMO
#   multiple ELMO layers
#   doc

from typing import Dict, Optional

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy
from typing import Dict, Optional, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import math
from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertLayerNorm, BertModel,
                                                BertPreTrainedModel, gelu)


import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BERTLikeLMHead(nn.Module):
    """BERT Head for masked language modeling."""

    def __init__(self, hidden_size, vocab_size):
        super(BERTLikeLMHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.dense.weight.data.normal_(mean=0.0, std=0.02)
        self.decoder.weight.data.normal_(mean=0.0, std=0.02)
        self.bias.data.zero_()
        self._dropout = torch.nn.Dropout(0.5)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self._dropout(x)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


class BERTLikeLMModelHead(nn.Module):

    def __init__(self, hidden_size, vocab_size):
        super(BERTLikeLMModelHead, self).__init__()
        self.lm_head = BERTLikeLMHead(hidden_size, vocab_size)

    def forward(self, sequence_output, choices_ids, gold_labels = None):
        prediction_scores = self.lm_head(sequence_output).gather(1, choices_ids)

        if gold_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores, gold_labels)

        return masked_lm_loss, prediction_scores

@Model.register("mlm_baseline")
class MLMBaseline(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.5,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        self._BERTLikeLMModelHead = BERTLikeLMModelHead(1000, vocab.get_vocab_size())

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        #initializer(self)

    def forward(self,  # type: ignore
                phrase: Dict[str, torch.LongTensor],
                choices: List[Dict[str, torch.LongTensor]],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        batch_size, num_of_tokens = phrase['tokens'].shape
        choices_ids = choices['tokens'][:,:,0].squeeze()
        _, num_choices = choices_ids.shape
        embedded_phrase = self._text_field_embedder(phrase)

        # putting the batch_size first, and concating the embeddings (the permute is to make
        # sure order is perserved
        embedded_phrase = embedded_phrase.view(batch_size, -1)
        # zero padding to reach the exact classifier size

        # now we expand to size (7, 11) by appending a row of 0s at pos 0 and pos 6,
        # and a column of 0s at pos 10
        embedded_phrase = F.pad(input=embedded_phrase, pad=(0, 1000 - num_of_tokens*50), mode='constant', value=0)

        # applying the 2 layes MLP
        loss, label_logits = self._BERTLikeLMModelHead(embedded_phrase, choices_ids, label)

        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        self._accuracy(label_logits, label)
        output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric = self._accuracy.get_metric(reset)
        return {
            'accuracy': metric,
            'EM': metric
        }

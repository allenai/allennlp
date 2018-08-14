"""
BiMPM (Bilateral Multi-Perspective Matching) model implementation.
"""

from typing import Dict, Optional, List, Any

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.modules.bimpm_matching import BiMPMMatching


@Model.register("bimpm")
class BiMPM(Model):
    """
    This ``Model`` implements BiMPM model described in `Bilateral Multi-Perspective Matching
    for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
    Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
    `PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    word_matcher : ``BiMPMMatching``
        BiMPM matching on the output of word embeddings of premise and hypothesis.
    encoder1 : ``Seq2SeqEncoder``
        First encoder layer for the premise and hypothesis
    matcher_fw1 : ``BiMPMMatching``
        BiMPM matching for the forward output of first encoder layer
    matcher_bw1 : ``BiMPMMatching``
        BiMPM matching for the backward output of first encoder layer
    encoder2 : ``Seq2SeqEncoder``
        Second encoder layer for the premise and hypothesis
    matcher_fw2 : ``BiMPMMatching``
        BiMPM matching for the forward output of second encoder layer
    matcher_bw2 : ``BiMPMMatching``
        BiMPM matching for the backward output of second encoder layer
    aggregator : ``Seq2VecEncoder``
        Aggregator of all BiMPM matching vectors
    classifier_feedforward : ``FeedForward``
        Fully connected layers for classification.
    dropout : ``float``
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_matcher: BiMPMMatching,
                 encoder1: Seq2SeqEncoder,
                 matcher_fw1: BiMPMMatching,
                 matcher_bw1: BiMPMMatching,
                 encoder2: Seq2SeqEncoder,
                 matcher_fw2: BiMPMMatching,
                 matcher_bw2: BiMPMMatching,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMPM, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder

        self.word_matcher = word_matcher

        self.encoder1 = encoder1
        self.matcher_fw1 = matcher_fw1
        self.matcher_bw1 = matcher_bw1

        self.encoder2 = encoder2
        self.matcher_fw2 = matcher_fw2
        self.matcher_bw2 = matcher_bw2

        self.aggregator = aggregator

        matching_dim = self.word_matcher.get_output_dim() + \
                       self.matcher_fw1.get_output_dim() + self.matcher_bw1.get_output_dim() + \
                       self.matcher_fw2.get_output_dim() + self.matcher_bw2.get_output_dim()

        if matching_dim != self.aggregator.get_input_dim():
            raise ConfigurationError("Matching dimension %d should match aggregator dimension %d" %
                                     (matching_dim, self.aggregator.get_input_dim()))

        self.classifier_feedforward = classifier_feedforward

        self.dropout = torch.nn.Dropout(dropout)

        self.metrics = {"accuracy": CategoricalAccuracy()}

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            The premise from a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            The hypothesis from a ``TextField``
        label : torch.LongTensor, optional (default = None)
            The label for the pair of the premise and the hypothesis
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information about the pair
        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        mask_p = util.get_text_field_mask(premise)
        mask_h = util.get_text_field_mask(hypothesis)

        def match_fw_bw(matcher_fw, matcher_bw, encoded_p_fw_bw, encoded_h_fw_bw):
            # The function to calculate matching vectors from both forward and backward
            # representations of the premise and the hypothesis
            dim = encoded_p_fw_bw.size(-1)
            assert dim == encoded_h_fw_bw.size(-1)
            encoded_p_fw, encoded_p_bw = torch.split(encoded_p_fw_bw, dim // 2, dim=-1)
            encoded_h_fw, encoded_h_bw = torch.split(encoded_h_fw_bw, dim // 2, dim=-1)
            mv_p_fw, mv_h_fw = matcher_fw(encoded_p_fw, mask_p, encoded_h_fw, mask_h)
            mv_p_bw, mv_h_bw = matcher_bw(encoded_p_bw, mask_p, encoded_h_bw, mask_h)

            return mv_p_fw, mv_h_fw, mv_p_bw, mv_h_bw

        # embedding and encoding of the premise
        embedded_p = self.dropout(self.text_field_embedder(premise))
        encoded_p1 = self.dropout(self.encoder1(embedded_p, mask_p))
        encoded_p2 = self.dropout(self.encoder2(encoded_p1, mask_p))

        # embedding and encoding of the hypothesis
        embedded_h = self.dropout(self.text_field_embedder(hypothesis))
        encoded_h1 = self.dropout(self.encoder1(embedded_h, mask_h))
        encoded_h2 = self.dropout(self.encoder2(encoded_h1, mask_h))

        # calculate matching vectors from word embedding, first layer encoding, and second layer encoding
        mv_word_p2h, mv_word_h2p = self.word_matcher(embedded_p, mask_p, embedded_h, mask_h)
        mv_p_fw1, mv_h_fw1, mv_p_bw1, mv_h_bw1 = \
            match_fw_bw(self.matcher_fw1, self.matcher_bw1, encoded_p1, encoded_h1)
        mv_p_fw2, mv_h_fw2, mv_p_bw2, mv_h_bw2 = \
            match_fw_bw(self.matcher_fw2, self.matcher_bw2, encoded_p2, encoded_h2)

        # concat the matching vectors
        mv_p = self.dropout(torch.cat(mv_word_p2h + mv_p_fw1 + mv_p_bw1 + mv_p_fw2 + mv_p_bw2, dim=2))
        mv_h = self.dropout(torch.cat(mv_word_h2p + mv_h_fw1 + mv_h_bw1 + mv_h_fw2 + mv_h_bw2, dim=2))

        # aggregate the matching vectors
        agg_p = self.dropout(self.aggregator(mv_p, mask_p))
        agg_h = self.dropout(self.aggregator(mv_h, mask_h))

        # the final forward layer
        logits = self.classifier_feedforward(torch.cat([agg_p, agg_h], dim=-1))

        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

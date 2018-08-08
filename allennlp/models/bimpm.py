"""
BiMPM (Bilateral Multi-Perspective Matching) model implementation.
Paper according to https://arxiv.org/pdf/1702.03814
Implementation according to https://github.com/zhiguowang/BiMPM/
"""

from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, MatchingLayer
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bimpm")
class BiMPM(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_matcher: MatchingLayer,
                 encoder1: Seq2SeqEncoder,
                 matcher_fw1: MatchingLayer,
                 matcher_bw1: MatchingLayer,
                 encoder2: Seq2SeqEncoder,
                 matcher_fw2: MatchingLayer,
                 matcher_bw2: MatchingLayer,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 dropout: float = 0.1,
                 num_perspective: int = 20,
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

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        mask_p = util.get_text_field_mask(premise)
        mask_h = util.get_text_field_mask(hypothesis)

        def match_fw_bw(matcher_fw, matcher_bw, encoded_p_fw_bw, encoded_h_fw_bw):
            dim = encoded_p_fw_bw.size(-1)
            assert dim == encoded_h_fw_bw.size(-1)
            encoded_p_fw, encoded_p_bw = torch.split(encoded_p_fw_bw, dim // 2, dim=-1)
            encoded_h_fw, encoded_h_bw = torch.split(encoded_h_fw_bw, dim // 2, dim=-1)
            mv_p_fw, mv_h_fw = matcher_fw(encoded_p_fw, mask_p, encoded_h_fw, mask_h)
            mv_p_bw, mv_h_bw = matcher_bw(encoded_p_bw, mask_p, encoded_h_bw, mask_h)

            return mv_p_fw, mv_h_fw, mv_p_bw, mv_h_bw

        embedded_p = self.dropout(self.text_field_embedder(premise))
        encoded_p1 = self.dropout(self.encoder1(embedded_p, mask_p))
        encoded_p2 = self.dropout(self.encoder2(encoded_p1, mask_p))

        embedded_h = self.dropout(self.text_field_embedder(hypothesis))
        encoded_h1 = self.dropout(self.encoder1(embedded_h, mask_h))
        encoded_h2 = self.dropout(self.encoder2(encoded_h1, mask_h))

        mv_word_p2h, mv_word_h2p = self.word_matcher(embedded_p, mask_p, embedded_h, mask_h)
        mv_p_fw1, mv_h_fw1, mv_p_bw1, mv_h_bw1 = match_fw_bw(self.matcher_fw1, self.matcher_bw1, encoded_p1, encoded_h1)
        mv_p_fw2, mv_h_fw2, mv_p_bw2, mv_h_bw2 = match_fw_bw(self.matcher_fw2, self.matcher_bw2, encoded_p2, encoded_h2)

        mv_p = self.dropout(torch.cat(mv_word_p2h + mv_p_fw1 + mv_p_bw1 + mv_p_fw2 + mv_p_bw2, dim=2))
        mv_h = self.dropout(torch.cat(mv_word_h2p + mv_h_fw1 + mv_h_bw1 + mv_h_fw2 + mv_h_bw2, dim=2))

        agg_p = self.dropout(self.aggregator(mv_p, mask_p))
        agg_h = self.dropout(self.aggregator(mv_h, mask_h))

        logits = self.classifier_feedforward(torch.cat([agg_p, agg_h], dim=-1))

        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


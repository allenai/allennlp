from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import (get_final_encoder_states, masked_max, masked_mean, masked_log_softmax)
from allennlp.common.checks import ConfigurationError


@Model.register("classifier")
class Classifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder = None,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 aggregations: List[str] = None,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab)
        self._input_embedder = input_embedder
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        if seq2vec_encoder:
            self._seq2vec_encoder = seq2vec_encoder
            self._clf_input_dim = self._seq2vec_encoder.get_output_dim()
            self._seq2seq_encoder = None
        elif seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
            self._clf_input_dim = self._seq2seq_encoder.get_output_dim() * len(aggregations)
            self._seq2vec_encoder = None
        else:
            self._seq2seq_encoder = None
            self._seq2vec_encoder = None
            self._clf_input_dim = self._input_embedder.get_output_dim()

        if aggregations:
            if "attention" in aggregations:
                if seq2vec_encoder:
                    encoder = self._seq2vec_encoder
                elif seq2seq_encoder:
                    encoder = self._seq2seq_encoder
                else:
                    encoder = self._input_embedder
                self._attention_layer = torch.nn.Linear(encoder.get_output_dim(), 1)
        
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._aggregations = aggregations
        self._classification_layer = torch.nn.Linear(self._clf_input_dim, self._num_labels)
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata to persist

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._input_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2vec_encoder:
            embedded_text = self._seq2vec_encoder(embedded_text,
                                                  mask=mask)
        elif self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text,
                                                  mask=mask)

        if self._aggregations:
            encoded_repr = []
            for aggregation in self._aggregations:
                if aggregation == 'maxpool':
                    broadcast_mask = mask.unsqueeze(-1).float()
                    encoded_text = embedded_text * broadcast_mask
                    encoded_text = masked_max(encoded_text,
                                               broadcast_mask,
                                               dim=1)
                elif aggregation == 'meanpool':
                    broadcast_mask = mask.unsqueeze(-1).float()
                    encoded_text = embedded_text * broadcast_mask
                    encoded_text = masked_mean(encoded_text,
                                                broadcast_mask,
                                                dim=1,
                                                keepdim=False)
                elif aggregation == 'final_state':
                    is_bi = self._encoder.is_bidirectional()
                    encoded_text = get_final_encoder_states(embedded_text,
                                                             mask,
                                                             is_bi)
                elif aggregation == 'attention':
                    alpha = self._attention_layer(embedded_text)
                    alpha = masked_log_softmax(alpha, mask.unsqueeze(-1), dim=1).exp()
                    encoded_text = alpha * embedded_text
                    encoded_text = encoded_text.sum(dim=1)
                else:
                    raise ConfigurationError(f"{aggregation} aggregation not available.")
                encoded_repr.append(encoded_text)
            embedded_text = torch.cat(encoded_repr, 1)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        label_logits = self._classification_layer(embedded_text)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(label_logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics

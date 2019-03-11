from typing import Any, Dict, List

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("basic_classifier")
class BasicClassifier(Model):
    """
    This ``Model`` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a Seq2SeqEncoder and/or
    a Seq2VecEncoder and then pass the embeddings to a linear classification layer,
    which projects into the label space. If neither Seq2SeqEncoder nor Seq2VecEncoder is
    provided, we will pass the embedded text directly to the linear classification layer.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the input text into a ``TextField``
    seq2seq_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : ``Seq2VecEnoder``, optional (default = ``None``)
        Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder will pool its output.
        Otherwise, this encoder will operate directly on the output of the `text_field_embedder`.
    dropout : ``float``, optional (default=None)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 seq2vec_encoder: Seq2VecEncoder = None,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:

        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        if seq2seq_encoder and not seq2vec_encoder:
            raise ConfigurationError("If you provide a seq2seq encoder,"
                                     "you must provide a seq2vec encoder to pool its output.")

        if seq2vec_encoder:
            self._seq2vec_encoder = seq2vec_encoder
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()
        else:
            self._seq2vec_encoder = None
            self._classifier_input_dim = self._text_field_embedder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
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

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        if self._seq2vec_encoder:
            embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics

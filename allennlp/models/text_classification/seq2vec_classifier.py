from typing import Dict, Optional, List, Any
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask


@Model.register("seq2vec_classifier")
class Seq2VecClassifier(Model):
    """
    This ``Model`` implements a Seq2Vec classifier.
    Look at allennlp.modules.seq2vec_encoders for available encoders.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``
        Used to encode the text
    output_feedforward : ``FeedForward``
        Used to prepare the text for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 output_feedforward: FeedForward,
                 classification_layer: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._encoder = encoder
        self._output_feedforward = output_feedforward
        self._classification_layer = classification_layer
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor=None,
                metadata: List[Dict[str, Any]]=None  # pylint:disable=unused-argument
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
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        encoded_text = self._encoder(embedded_text, mask)

        if self.dropout is not None:
            encoded_text = self.dropout(encoded_text)

        output_hidden = self._output_feedforward(encoded_text)
        label_logits = self._classification_layer(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}
        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            output_dict["loss"] = loss

        self._accuracy(label_logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

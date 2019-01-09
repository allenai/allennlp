from typing import Dict, Optional, List, Any
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("logistic_regression")
class LogisticRegression(Model):
    """
    This ``Model`` implements a basic logistic regression classifier
    on a onehot embedding of text.

    Parameters
    ----------
    vocab : ``Vocabulary``
    onehot_embedder : ``TextFieldEmbedder``
        Used to embed the ``TextField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self,
                 vocab: Vocabulary,
                 onehot_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = onehot_embedder

        self._vocab_size = vocab.get_vocab_size(namespace="tokens")
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._classification_layer = torch.nn.Linear(self._vocab_size,
                                                     self._num_labels)
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
            Metadata on tokens to persist

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
        # generate onehot bag of words embeddings
        embedded_text = self._text_field_embedder(tokens)
        linear_output = self._classification_layer(embedded_text)
        label_probs = torch.nn.functional.log_softmax(linear_output, dim=-1)
        output_dict = {"label_logits": linear_output, "label_probs": label_probs}
        if label is not None:
            loss = self._loss(linear_output, label.long().view(-1))
            output_dict["loss"] = loss
        self._accuracy(linear_output, label)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

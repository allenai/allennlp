from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.heads.head import Head
from allennlp.modules import FeedForward, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy


@Head.register("classifier")
class ClassifierHead(Head):
    """
    A classification `Head`.  Takes encoded text, gets a single vector out of it, runs an optional
    feedforward layer on that vector, then classifies it into some label space.

    Registered as a `Head` with name "classifier".

    # Parameters

    vocab : `Vocabulary`
        Used to get the number of labels, if `num_labels` is not provided, and to translate label
        indices to strings in `make_output_human_readable`.
    seq2vec_encoder : `Seq2VecEncoder`
        The input to this module is assumed to be a sequence of encoded vectors.  We use a
        `Seq2VecEncoder` to compress this into a single vector on which we can perform
        classification.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply on the pooled output before performing the
        classification.
    input_dim : `int`, optional (default = `None`)
        We need to know how many dimensions to use for the final classification weight matrix.  If
        you have provided either a `seq2vec_encoder` or a `feedforward` module, we can get the
        correct size from those objects.  If you use default values for both of those parameters,
        then you must provide this parameter, so that we know the size of that encoding.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = `"labels"`)
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
        input_dim: int = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
    ) -> None:

        super().__init__(vocab)
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if self._feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim() or input_dim

        if self._classifier_input_dim is None:
            raise ValueError("No input dimension given!")

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self,
        encoded_text: torch.FloatTensor,
        encoded_text_mask: torch.BoolTensor,
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        encoding = self._seq2vec_encoder(encoded_text, mask=encoded_text_mask)

        if self._dropout:
            encoding = self._dropout(encoding)

        if self._feedforward is not None:
            encoding = self._feedforward(encoding)

        logits = self._classification_layer(encoding)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        if "probs" in output_dict:
            predictions = output_dict["probs"]
            if predictions.dim() == 2:
                predictions_list = [predictions[i] for i in range(predictions.shape[0])]
            else:
                predictions_list = [predictions]
            classes = []
            for prediction in predictions_list:
                label_idx = prediction.argmax(dim=-1).item()
                label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                    label_idx, str(label_idx)
                )
                classes.append(label_str)
            output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

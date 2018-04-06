from typing import Dict, Optional, Union

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, Maxout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bcn")
class BiattentiveClassificationNetwork(Model):
    """
    This class implements the Biattentive Classification Network model described
    in section 5 of `Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    <https://arxiv.org/abs/1708.00107>`_ for text classification. We assume we're
    given a piece of text, and we predict some output label.

    At a high level, the model starts by embedding the tokens and running them through
    a feed-forward neural net (``pre_encode_feedforward``). Then, we encode these
    representations with a ``Seq2SeqEncoder`` (``encoder``). We run biattention
    on the encoder output represenatations (self-attention in this case, since
    the two representations that typically go into biattention are identical) and
    get out an attentive vector representation of the text. We combine this text
    representation with the encoder outputs computed earlier, and then run this through
    yet another ``Seq2SeqEncoder`` (the ``integrator``). Lastly, we take the output of the
    integrator and max, min, mean, and self-attention pool to create a final representation,
    which is passed through a maxout network or some feed-forward layers
    to output a classification (``output_layer``).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    embedding_dropout : ``float``
        The amount of dropout to apply on the embeddings.
    pre_encode_feedforward : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    integrator : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the token encodings.
    integrator_dropout : ``float``
        The amount of dropout to apply on integrator output.
    output_layer : ``Union[Maxout, FeedForward]``
        The maxout or feed forward network that takes the final representations and produces
        a classification prediction.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 pre_encode_feedforward: FeedForward,
                 encoder: Seq2SeqEncoder,
                 integrator: Seq2SeqEncoder,
                 integrator_dropout: float,
                 output_layer: Union[FeedForward, Maxout],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiattentiveClassificationNetwork, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._integrator = integrator
        self._self_attentive_pooling_projection = nn.Linear(
                self._integrator.get_output_dim(), 1)
        self._output_layer = output_layer
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._integrator_dropout = nn.Dropout(integrator_dropout)

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               self._pre_encode_feedforward.get_input_dim(),
                               "text field embedder output dim",
                               "Pre-encoder feedforward input dim")
        check_dimensions_match(self._pre_encode_feedforward.get_output_dim(),
                               self._encoder.get_input_dim(),
                               "Pre-encoder feedforward output dim",
                               "Encoder input dim")
        check_dimensions_match(self._encoder.get_output_dim() * 3,
                               self._integrator.get_input_dim(),
                               "Encoder output dim * 3",
                               "Integrator input dim")
        check_dimensions_match(self._integrator.get_output_dim() * 4,
                               self._output_layer.get_input_dim(),
                               "Integrator output dim * 4",
                               "Output layer input dim")
        check_dimensions_match(self._output_layer.get_output_dim(),
                               self._num_classes,
                               "Output layer output dim",
                               "Number of classes.")

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        text_mask = util.get_text_field_mask(tokens).float()
        embedded_text = self._text_field_embedder(tokens)
        dropped_embedded_text = self._embedding_dropout(embedded_text)

        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)
        encoded_tokens = self._encoder(pre_encoded_text, text_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.last_dim_softmax(attention_logits, text_mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens,
                                      encoded_tokens - encoded_text,
                                      encoded_tokens * encoded_text], 2)
        integrated_encodings = self._integrator(integrator_input, text_mask)

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
                integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)

        logits = self._output_layer(pooled_representations_dropped)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BiattentiveClassificationNetwork':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        embedding_dropout = params.pop("embedding_dropout")
        pre_encode_feedforward = FeedForward.from_params(params.pop("pre_encode_feedforward"))
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        integrator = Seq2SeqEncoder.from_params(params.pop("integrator"))
        integrator_dropout = params.pop("integrator_dropout")

        output_layer_params = params.pop("output_layer")
        if "activations" in output_layer_params:
            output_layer = FeedForward.from_params(output_layer_params)
        else:
            output_layer = Maxout.from_params(output_layer_params)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   embedding_dropout=embedding_dropout,
                   pre_encode_feedforward=pre_encode_feedforward,
                   encoder=encoder,
                   integrator=integrator,
                   integrator_dropout=integrator_dropout,
                   output_layer=output_layer,
                   initializer=initializer,
                   regularizer=regularizer)

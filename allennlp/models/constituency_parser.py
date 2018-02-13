from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("constituency_parser")
class SpanConstituencyParser(Model):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text 
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a 
    ``SpanExtractor``, and then predicts a tag for each span in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    span_extractor : ``SpanExtractor``, required.
        The method used to extract the spans from the encoded sequence.
    stacked_encoder : ``Seq2SeqEncoder``, required.
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and generating span representations.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 span_extractor: SpanExtractor,
                 stacked_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleTagger, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.span_extractor = span_extractor
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.stacked_encoder = stacked_encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.span_extractor.get_output_dim(),
                                                           self.num_classes))

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               stacked_encoder.get_input_dim(),
                               "text field embedding dim",
                               "encoder input dim")
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                spans: torch.LongTensor,
                span_labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        spans : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)`` representing the
            inclusive start and end indices of all possible spans in the sentence.
        span_labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape ``(batch_size, num_spans)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing unnormalised log probabilities of the tag classes.
        class_probabilities : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution of the tag classes per span.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        num_spans = spans.size(1)
        mask = get_text_field_mask(tokens)
        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()

        # TODO(Mark): add masks once other PR is merged.
        encoded_text = self.stacked_encoder(embedded_text_input, mask)
        span_representations = self.span_extractor(encoded_text, spans)
        
        logits = self.tag_projection_layer(span_representations)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          num_spans,
                                                                          self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, span_labels, mask)
            for metric in self.metrics.values():
                metric(logits, span_labels, mask)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            
            # TODO(Mark): construct the ML tree from the scored spans.
            pass            

        output_dict['tags'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SpanConstituencyParser':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        span_extractor = SpanExtractor.from_params(params.pop("span_extractor"))
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   span_extractor=span_extractor,
                   stacked_encoder=stacked_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
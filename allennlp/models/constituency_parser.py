from typing import Dict, Tuple, List, Optional

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import last_dim_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("constituency_parser")
class SpanConstituencyParser(Model):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a
    ``SpanExtractor``, and then predicts a label for each span in the sequence.

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
        super(SpanConstituencyParser, self).__init__(vocab, regularizer)

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
                pos_tags: torch.LongTensor,
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
            representing unnormalised log probabilities of the label classes for each span.
        class_probabilities : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        # TODO(Mark): merge this into the call to the span extractor once other PR is in.
        spans = spans * span_mask.long().unsqueeze(-1)

        encoded_text = self.stacked_encoder(embedded_text_input, mask)
        # TODO(Mark): add masks once other PR is merged.
        span_representations = self.span_extractor(encoded_text, spans)
        logits = self.tag_projection_layer(span_representations)
        class_probabilities = last_dim_softmax(logits, span_mask.unsqueeze(-1))
        output_dict = {
                "logits": logits,
                "class_probabilities": class_probabilities,
                "spans": spans,
                "tokens": tokens,
                "token_mask": mask
        }
        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
            for metric in self.metrics.values():
                metric(logits, span_labels, span_mask)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constructs a tree given the scored spans. Spans which are predicted to have no label
        are not included.
        """
        all_predictions = output_dict['class_probabilities'].cpu().data
        all_spans = output_dict["spans"].cpu().data
        no_label_id = self.vocab.get_token_index("NO-LABEL", "labels")

        all_sentences = output_dict["tokens"]["tokens"].data
        sentence_lengths = get_lengths_from_binary_sequence_mask(output_dict["token_mask"]).data

        trees = []
        for batch_index, (predictions, spans, sentence) in enumerate(zip(all_predictions,
                                                                         all_spans,
                                                                         all_sentences)):
            sentence: List[str] = [self.vocab.get_token_from_index(index, "tokens") for
                                   index in sentence[:sentence_lengths[batch_index]]]

            selected_spans = []
            for prediction, span in zip(predictions, spans):
                start, end = span
                no_label_prob = prediction[no_label_id]
                label_prob, label_index = torch.max(prediction, -1)

                # Does the span have a label != NO-LABEL or is it the root node?
                # If so, include it in the spans that we consider.
                if int(label_index) != no_label_id or (start == 0 and end + 1 == len(sentence)):
                    selected_spans.append({
                            "start": int(start),
                            # Switch to exclusive span ends to make
                            # recursive tree constuction easier.
                            "end": int(end) + 1,
                            "label_prob": float(label_prob),
                            "no_label_prob": float(no_label_prob),
                            "label_index": int(label_index)
                    })

            # The spans we've selected might overlap, which causes problems when we try
            # to construct the tree as they won't nest properly.
            consistent_spans = self.resolve_overlap_conflicts_greedily(selected_spans)

            spans_to_labels = {(span["start"], span["end"]): self.vocab.get_token_from_index(span["label_index"], "labels")
                               for span in consistent_spans}
            trees.append(self.construct_tree_from_spans(spans_to_labels, sentence))

        output_dict["trees"] = trees
        return output_dict

    @staticmethod
    def resolve_overlap_conflicts_greedily(chosen_spans: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        Given a set of spans, removes spans which overlap.
        """
        conflicts_exist = True
        while conflicts_exist:
            conflicts_exist = False
            for span1_index, span1 in enumerate(chosen_spans):
                for span2_index, span2 in list(enumerate(chosen_spans))[span1_index + 1:]:
                    if (span1["start"] < span2["start"] < span1["end"] < span2["end"] or
                        span2["start"] < span1["start"] < span2["end"] < span1["end"]):
                        conflicts_exist = True
                        if (span1["no_label_prob"] + span2["label_prob"] <
                            span2["no_label_prob"] + span1["label_prob"]):
                            chosen_spans.pop(span2_index)
                        else:
                            chosen_spans.pop(span1_index)
                        break
        return chosen_spans
    @staticmethod
    def construct_tree_from_spans(spans_to_labels: Dict[Tuple[int, int], str],
                                  sentence: List[str],
                                  pos_tags: List[str] = None):

        def assemble_subtree(start, end):
            if (start, end) in spans_to_labels:
                label = spans_to_labels[(start, end)]
                assert label != ()
            else:
                assert start != 0 or end != len(sentence)
                label = None

            if end - start == 1:
                word = sentence[start]
                tree = {"start": start, "word": word, "is_leaf": True}
                if label is not None:
                    tree["label"] = label
                if pos_tags is not None:
                    tree["pos_tag"] = pos_tags[start]
                return [tree]

            argmax_split = start + 1
            # Find the next largest subspan such that
            # the left hand side is a constituent.
            for split in range(end - 1, start, -1):
                if (start, split) in spans_to_labels:
                    argmax_split = split
                    break

            assert start < argmax_split < end, (start, argmax_split, end)
            left_trees = assemble_subtree(start, argmax_split)
            right_trees = assemble_subtree(argmax_split, end)
            children = left_trees + right_trees
            if label is not None:
                children = [{"label": label, "children": children, "start": start, "end": end}]
            return children

        tree = assemble_subtree(0, len(sentence))
        return tree[0]

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

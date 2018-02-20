from typing import Dict, Tuple, List, Optional, NamedTuple
from overrides import overrides

import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import last_dim_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import F1Measure
from allennlp.training.metrics import EvalbBracketingScorer


class SpanInformation(NamedTuple):
    """
    A helper namedtuple for handling decoding information.

    Parameters
    ----------
    start : ``int``
        The start index of the span.
    end : ``int``
        The exclusive end index of the span.
    no_label_prob : ``float``
        The probability of this span being assigned the ``NO-LABEL`` label.
    label_prob : ``float``
        The probability of the most likely label.
    """
    start: int
    end: int
    label_prob: float
    no_label_prob: float
    label_index: int


@Model.register("constituency_parser")
class SpanConstituencyParser(Model):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a
    ``SpanExtractor``, and then predicts a label for each span in the sequence.
    These labels are non-terminal nodes in a constituency parse tree, which we then
    greedily reconstruct.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    span_extractor : ``SpanExtractor``, required.
        The method used to extract the spans from the encoded sequence.
    encoder : ``Seq2SeqEncoder``, required.
        The encoder that we will use in between embedding tokens and
        generating span representations.
    feedforward_layer : ``FeedForward``, required.
        The FeedForward layer that we will use in between the encoder and the linear
        projection to a distribution over span labels.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 span_extractor: SpanExtractor,
                 encoder: Seq2SeqEncoder,
                 feedforward_layer: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 evalb_directory_path: str = None) -> None:
        super(SpanConstituencyParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.span_extractor = span_extractor
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.feedforward_layer = TimeDistributed(feedforward_layer) if feedforward_layer else None

        if feedforward_layer is not None:
            output_dim = feedforward_layer.get_output_dim()
        else:
            output_dim = span_extractor.get_output_dim()

        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_classes))

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               encoder.get_input_dim(),
                               "text field embedding dim",
                               "encoder input dim")
        if feedforward_layer is not None:
            check_dimensions_match(encoder.get_output_dim(),
                                   feedforward_layer.get_input_dim(),
                                   "stacked encoder output dim",
                                   "feedforward input dim")

        self.metrics = {label: F1Measure(index) for index, label
                        in self.vocab.get_index_to_token_vocabulary("labels").items()}

        if evalb_directory_path is not None:
            self._evalb_score = EvalbBracketingScorer(evalb_directory_path)
        else:
            self._evalb_score = None
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                spans: torch.LongTensor,
                span_labels: torch.LongTensor = None,
                gold_tree: List[Tree] = None) -> Dict[str, torch.Tensor]:
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
        span_labels : ``torch.LongTensor``, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape ``(batch_size, num_spans)``.
        gold_tree : ``List[Tree]``, optional, (default = None)
            Gold NLTK trees for use in evaluation.

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
        sentence_lengths = get_lengths_from_binary_sequence_mask(mask)
        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()

        encoded_text = self.encoder(embedded_text_input, mask)
        span_representations = self.span_extractor(encoded_text, spans, mask, span_mask)
        if self.feedforward_layer is not None:
            span_representations = self.feedforward_layer(span_representations)
        logits = self.tag_projection_layer(span_representations)
        class_probabilities = last_dim_softmax(logits, span_mask.unsqueeze(-1))

        output_dict = {
                "class_probabilities": class_probabilities,
                "spans": spans,
                # TODO(Mark): This relies on having tokens represented with a SingleIdTokenIndexer...
                "tokens": tokens["tokens"],
                "sentence_lengths": sentence_lengths
        }
        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
            for metric in self.metrics.values():
                metric(logits, span_labels, span_mask)
            output_dict["loss"] = loss

        # The evalb score is expensive to compute, so we only compute
        # it for the validation and test sets.
        if gold_tree is not None and self._evalb_score is not None and not self.training:
            predicted_trees = self.construct_trees(class_probabilities.cpu().data,
                                                   spans.cpu().data,
                                                   tokens["tokens"].cpu().data,
                                                   sentence_lengths.cpu().data)
            self._evalb_score(predicted_trees, gold_tree)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constructs an NLTK ``Tree`` given the scored spans. We also switch to exclusive
        span ends when constructing the tree representation, because it makes indexing
        into lists cleaner for ranges of text, rather than individual indices.
        """
        all_predictions = output_dict['class_probabilities'].cpu().data
        all_spans = output_dict["spans"].cpu().data

        all_sentences = output_dict["tokens"].cpu().data
        sentence_lengths = output_dict["sentence_lengths"].data
        trees = self.construct_trees(all_predictions, all_spans, all_sentences, sentence_lengths)

        output_dict["trees"] = trees
        return output_dict

    def construct_trees(self,
                        predictions: torch.FloatTensor,
                        enumerated_spans: torch.LongTensor,
                        sentences: torch.LongTensor,
                        sentence_lengths: torch.LongTensor) -> List[Tree]:
        """
        Construct ``nltk.Tree``'s for each batch element by greedily nesting spans.
        The trees use exclusive end indices, which contrasts with how spans are
        represented in the rest of the model.
        Parameters
        ----------

        predictions : ``torch.FloatTensor``, required.
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        enumerated_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the span
            indices we scored.
        sentences : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, sentence_length) representing the vocabulary
            ids of the words in the sentences.
        sentence_lengths : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of the non-padded
            elements of ``sentences``.

        Returns
        -------
        A ``List[Tree]`` containing the decoded trees for each element in the batch.
        """
        # Switch to using exclusive end spans.
        exclusive_end_spans = enumerated_spans.clone()
        exclusive_end_spans[:, :, -1] += 1
        no_label_id = self.vocab.get_token_index("NO-LABEL", "labels")

        trees: List[Tree] = []
        for batch_index, (scored_spans, spans, sentence_ids) in enumerate(zip(predictions,
                                                                              exclusive_end_spans,
                                                                              sentences)):
            sentence: List[str] = [self.vocab.get_token_from_index(index, "tokens") for
                                   index in sentence_ids[:sentence_lengths[batch_index]]]

            selected_spans = []
            for prediction, span in zip(scored_spans, spans):
                start, end = span
                no_label_prob = prediction[no_label_id]
                label_prob, label_index = torch.max(prediction, -1)

                # Does the span have a label != NO-LABEL or is it the root node?
                # If so, include it in the spans that we consider.
                if int(label_index) != no_label_id or (start == 0 and end == len(sentence)):
                    # TODO(Mark): Remove this once pylint sorts out named tuples.
                    # https://github.com/PyCQA/pylint/issues/1418
                    selected_spans.append(SpanInformation(start=int(start), # pylint: disable=no-value-for-parameter
                                                          end=int(end),
                                                          label_prob=float(label_prob),
                                                          no_label_prob=float(no_label_prob),
                                                          label_index=int(label_index)))

            # The spans we've selected might overlap, which causes problems when we try
            # to construct the tree as they won't nest properly.
            consistent_spans = self.resolve_overlap_conflicts_greedily(selected_spans)

            spans_to_labels = {(span.start, span.end):
                               self.vocab.get_token_from_index(span.label_index, "labels")
                               for span in consistent_spans}
            trees.append(self.construct_tree_from_spans(spans_to_labels, sentence))

        return trees

    @staticmethod
    def resolve_overlap_conflicts_greedily(spans: List[SpanInformation]) -> List[SpanInformation]:
        """
        Given a set of spans, removes spans which overlap by evaluating the difference
        in probability between one being labeled and the other explicitly having no label
        and vice-versa. The worst case time complexity of this method is ``O(k * n^4)`` where ``n``
        is the length of the sentence that the spans were enumerated from (and therefore
        ``k * m^2`` complexity with respect to the number of spans ``m``) and ``k`` is the
        number of conflicts. However, in practice, there are very few conflicts. Hopefully.

        This function modifies ``spans`` to remove overlapping spans.

        Parameters
        ----------
        spans: ``List[SpanInformation]``, required.
            A list of spans, where each span is a ``namedtuple`` containing the
            following attributes:

        start : ``int``
            The start index of the span.
        end : ``int``
            The exclusive end index of the span.
        no_label_prob : ``float``
            The probability of this span being assigned the ``NO-LABEL`` label.
        label_prob : ``float``
            The probability of the most likely label.

        Returns
        -------
        A modified list of ``spans``, with the conflicts resolved by considering local
        differences between pairs of spans and removing one of the two spans.
        """
        conflicts_exist = True
        while conflicts_exist:
            conflicts_exist = False
            for span1_index, span1 in enumerate(spans):
                for span2_index, span2 in list(enumerate(spans))[span1_index + 1:]:
                    if (span1.start < span2.start < span1.end < span2.end or
                                span2.start < span1.start < span2.end < span1.end):
                        # The spans overlap.
                        conflicts_exist = True
                        # What's the more likely situation: that span2 was labeled
                        # and span1 was unlabled, or that span1 was labeled and span2
                        # was unlabled? In the first case, we delete span2 from the
                        # set of spans to form the tree - in the second case, we delete
                        # span1.
                        if (span1.no_label_prob + span2.label_prob <
                                    span2.no_label_prob + span1.label_prob):
                            spans.pop(span2_index)
                        else:
                            spans.pop(span1_index)
                        break
        return spans

    @staticmethod
    def construct_tree_from_spans(spans_to_labels: Dict[Tuple[int, int], str],
                                  sentence: List[str],
                                  pos_tags: List[str] = None) -> Tree:
        """
        Parameters
        ----------
        spans_to_labels : ``Dict[Tuple[int, int], str]``, required.
            A mapping from spans to constituency labels.
        sentence : ``List[str]``, required.
            A list of tokens forming the sentence to be parsed.
        pos_tags : ``List[str]``, optional (default = None)
            A list of the pos tags for the words in the sentence, if they
            were either predicted or taken as input to the model.

        Returns
        -------
        An ``nltk.Tree`` constructed from the labelled spans.
        """
        def assemble_subtree(start: int, end: int):
            if (start, end) in spans_to_labels:
                label = spans_to_labels[(start, end)]
            else:
                label = None

            # This node is a leaf.
            if end - start == 1:
                word = sentence[start]
                pos_tag = pos_tags[start] if pos_tags is not None else "XX"
                tree = Tree(pos_tag, [word])
                if label is not None and pos_tags is not None:
                    # If POS tags were passed explicitly,
                    # they are added as pre-terminal nodes.
                    tree = Tree(label, [tree])
                elif label is not None:
                    # Otherwise, we didn't want POS tags
                    # at all.
                    tree = Tree(label, [word])
                return [tree]

            argmax_split = start + 1
            # Find the next largest subspan such that
            # the left hand side is a constituent.
            for split in range(end - 1, start, -1):
                if (start, split) in spans_to_labels:
                    argmax_split = split
                    break

            left_trees = assemble_subtree(start, argmax_split)
            right_trees = assemble_subtree(argmax_split, end)
            children = left_trees + right_trees
            if label is not None:
                children = [Tree(label, children)]
            return children

        tree = assemble_subtree(0, len(sentence))
        return tree[0]

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self.metrics.items():
            f1, precision, recall = metric.get_metric(reset) # pylint: disable=invalid-name
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            all_metrics[metric_name + "_f1"] = f1
            all_metrics[metric_name + "_precision"] = precision
            all_metrics[metric_name + "_recall"] = recall

        num_metrics = len(self.metrics)
        all_metrics["average_f1"] = total_f1 / num_metrics
        all_metrics["average_precision"] = total_precision / num_metrics
        all_metrics["average_recall"] = total_recall / num_metrics

        if self._evalb_score is not None:
            evalb_metrics = self._evalb_score.get_metric(reset=reset)
            all_metrics.update(evalb_metrics)

        return all_metrics

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SpanConstituencyParser':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        span_extractor = SpanExtractor.from_params(params.pop("span_extractor"))
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        feed_forward_params = params.pop("feedforward", None)
        if feed_forward_params is not None:
            feedforward_layer = FeedForward.from_params(feed_forward_params)
        else:
            feedforward_layer = None
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        evalb_directory_path = params.pop("evalb_directory_path", None)
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   span_extractor=span_extractor,
                   encoder=encoder,
                   feedforward_layer=feedforward_layer,
                   initializer=initializer,
                   regularizer=regularizer,
                   evalb_directory_path=evalb_directory_path)

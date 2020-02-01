from typing import Dict, Tuple, List, NamedTuple, Any
from overrides import overrides

import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import EvalbBracketingScorer, DEFAULT_EVALB_DIR
from allennlp.common.checks import ConfigurationError


class SpanInformation(NamedTuple):
    """
    A helper namedtuple for handling decoding information.

    # Parameters

    start : `int`
        The start index of the span.
    end : `int`
        The exclusive end index of the span.
    no_label_prob : `float`
        The probability of this span being assigned the `NO-LABEL` label.
    label_prob : `float`
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
    This `SpanConstituencyParser` simply encodes a sequence of text
    with a stacked `Seq2SeqEncoder`, extracts span representations using a
    `SpanExtractor`, and then predicts a label for each span in the sequence.
    These labels are non-terminal nodes in a constituency parse tree, which we then
    greedily reconstruct.

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    span_extractor : `SpanExtractor`, required.
        The method used to extract the spans from the encoded sequence.
    encoder : `Seq2SeqEncoder`, required.
        The encoder that we will use in between embedding tokens and
        generating span representations.
    feedforward : `FeedForward`, required.
        The FeedForward layer that we will use in between the encoder and the linear
        projection to a distribution over span labels.
    pos_tag_embedding : `Embedding`, optional.
        Used to embed the `pos_tags` `SequenceLabelField` we get as input to the model.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    evalb_directory_path : `str`, optional (default=`DEFAULT_EVALB_DIR`)
        The path to the directory containing the EVALB executable used to score
        bracketed parses. By default, will use the EVALB included with allennlp,
        which is located at allennlp/tools/EVALB . If `None`, EVALB scoring
        is not used.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        span_extractor: SpanExtractor,
        encoder: Seq2SeqEncoder,
        feedforward: FeedForward = None,
        pos_tag_embedding: Embedding = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        evalb_directory_path: str = DEFAULT_EVALB_DIR,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.span_extractor = span_extractor
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.feedforward_layer = TimeDistributed(feedforward) if feedforward else None
        self.pos_tag_embedding = pos_tag_embedding or None
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = span_extractor.get_output_dim()

        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_classes))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()
        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "representation dim (tokens + optional POS tags)",
            "encoder input dim",
        )
        check_dimensions_match(
            encoder.get_output_dim(),
            span_extractor.get_input_dim(),
            "encoder input dim",
            "span extractor input dim",
        )
        if feedforward is not None:
            check_dimensions_match(
                span_extractor.get_output_dim(),
                feedforward.get_input_dim(),
                "span extractor output dim",
                "feedforward input dim",
            )

        self.tag_accuracy = CategoricalAccuracy()

        if evalb_directory_path is not None:
            self._evalb_score = EvalbBracketingScorer(evalb_directory_path)
        else:
            self._evalb_score = None
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        spans: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        pos_tags: TextFieldTensors = None,
        span_labels: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : TextFieldTensors, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        spans : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)` representing the
            inclusive start and end indices of all possible spans in the sentence.
        metadata : List[Dict[str, Any]], required.
            A dictionary of metadata for each batch element which has keys:
                tokens : `List[str]`, required.
                    The original string tokens in the sentence.
                gold_tree : `nltk.Tree`, optional (default = None)
                    Gold NLTK trees for use in evaluation.
                pos_tags : `List[str]`, optional.
                    The POS tags for the sentence. These can be used in the
                    model as embedded features, but they are passed here
                    in addition for use in constructing the tree.
        pos_tags : `torch.LongTensor`, optional (default = None)
            The output of a `SequenceLabelField` containing POS tags.
        span_labels : `torch.LongTensor`, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape `(batch_size, num_spans)`.

        # Returns

        An output dictionary consisting of:
        class_probabilities : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_spans, span_label_vocab_size)`
            representing a distribution over the label classes per span.
        spans : `torch.LongTensor`
            The original spans tensor.
        tokens : `List[List[str]]`, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : `List[List[str]]`, required.
            A list of POS tags in the sentence for each element in the batch.
        num_spans : `torch.LongTensor`, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in `enumerated_spans`.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self.pos_tag_embedding is not None:
            embedded_pos_tags = self.pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self.pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)
        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()
        if span_mask.dim() == 1:
            # This happens if you use batch_size 1 and encounter
            # a length 1 sentence in PTB, which do exist. -.-
            span_mask = span_mask.unsqueeze(-1)
        if span_labels is not None and span_labels.dim() == 1:
            span_labels = span_labels.unsqueeze(-1)

        num_spans = get_lengths_from_binary_sequence_mask(span_mask)

        encoded_text = self.encoder(embedded_text_input, mask)

        span_representations = self.span_extractor(encoded_text, spans, mask, span_mask)

        if self.feedforward_layer is not None:
            span_representations = self.feedforward_layer(span_representations)

        logits = self.tag_projection_layer(span_representations)
        class_probabilities = masked_softmax(logits, span_mask.unsqueeze(-1))

        output_dict = {
            "class_probabilities": class_probabilities,
            "spans": spans,
            "tokens": [meta["tokens"] for meta in metadata],
            "pos_tags": [meta.get("pos_tags") for meta in metadata],
            "num_spans": num_spans,
        }
        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
            self.tag_accuracy(class_probabilities, span_labels, span_mask)
            output_dict["loss"] = loss

        # The evalb score is expensive to compute, so we only compute
        # it for the validation and test sets.
        batch_gold_trees = [meta.get("gold_tree") for meta in metadata]
        if all(batch_gold_trees) and self._evalb_score is not None and not self.training:
            gold_pos_tags: List[List[str]] = [
                list(zip(*tree.pos()))[1] for tree in batch_gold_trees
            ]
            predicted_trees = self.construct_trees(
                class_probabilities.cpu().data,
                spans.cpu().data,
                num_spans.data,
                output_dict["tokens"],
                gold_pos_tags,
            )
            self._evalb_score(predicted_trees, batch_gold_trees)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constructs an NLTK `Tree` given the scored spans. We also switch to exclusive
        span ends when constructing the tree representation, because it makes indexing
        into lists cleaner for ranges of text, rather than individual indices.

        Finally, for batch prediction, we will have padded spans and class probabilities.
        In order to make this less confusing, we remove all the padded spans and
        distributions from `spans` and `class_probabilities` respectively.
        """
        all_predictions = output_dict["class_probabilities"].cpu().data
        all_spans = output_dict["spans"].cpu().data

        all_sentences = output_dict["tokens"]
        all_pos_tags = output_dict["pos_tags"] if all(output_dict["pos_tags"]) else None
        num_spans = output_dict["num_spans"].data
        trees = self.construct_trees(
            all_predictions, all_spans, num_spans, all_sentences, all_pos_tags
        )

        batch_size = all_predictions.size(0)
        output_dict["spans"] = [all_spans[i, : num_spans[i]] for i in range(batch_size)]
        output_dict["class_probabilities"] = [
            all_predictions[i, : num_spans[i], :] for i in range(batch_size)
        ]

        output_dict["trees"] = trees
        return output_dict

    def construct_trees(
        self,
        predictions: torch.FloatTensor,
        all_spans: torch.LongTensor,
        num_spans: torch.LongTensor,
        sentences: List[List[str]],
        pos_tags: List[List[str]] = None,
    ) -> List[Tree]:
        """
        Construct `nltk.Tree`'s for each batch element by greedily nesting spans.
        The trees use exclusive end indices, which contrasts with how spans are
        represented in the rest of the model.

        # Parameters

        predictions : `torch.FloatTensor`, required.
            A tensor of shape `(batch_size, num_spans, span_label_vocab_size)`
            representing a distribution over the label classes per span.
        all_spans : `torch.LongTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the span
            indices we scored.
        num_spans : `torch.LongTensor`, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in `enumerated_spans`.
        sentences : `List[List[str]]`, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : `List[List[str]]`, optional (default = None).
            A list of POS tags for each word in the sentence for each element
            in the batch.

        # Returns

        A `List[Tree]` containing the decoded trees for each element in the batch.
        """
        # Switch to using exclusive end spans.
        exclusive_end_spans = all_spans.clone()
        exclusive_end_spans[:, :, -1] += 1
        no_label_id = self.vocab.get_token_index("NO-LABEL", "labels")

        trees: List[Tree] = []
        for batch_index, (scored_spans, spans, sentence) in enumerate(
            zip(predictions, exclusive_end_spans, sentences)
        ):
            selected_spans = []
            for prediction, span in zip(
                scored_spans[: num_spans[batch_index]], spans[: num_spans[batch_index]]
            ):
                start, end = span
                no_label_prob = prediction[no_label_id]
                label_prob, label_index = torch.max(prediction, -1)

                # Does the span have a label != NO-LABEL or is it the root node?
                # If so, include it in the spans that we consider.
                if int(label_index) != no_label_id or (start == 0 and end == len(sentence)):
                    selected_spans.append(
                        SpanInformation(
                            start=int(start),
                            end=int(end),
                            label_prob=float(label_prob),
                            no_label_prob=float(no_label_prob),
                            label_index=int(label_index),
                        )
                    )

            # The spans we've selected might overlap, which causes problems when we try
            # to construct the tree as they won't nest properly.
            consistent_spans = self.resolve_overlap_conflicts_greedily(selected_spans)

            spans_to_labels = {
                (span.start, span.end): self.vocab.get_token_from_index(span.label_index, "labels")
                for span in consistent_spans
            }
            sentence_pos = pos_tags[batch_index] if pos_tags is not None else None
            trees.append(self.construct_tree_from_spans(spans_to_labels, sentence, sentence_pos))

        return trees

    @staticmethod
    def resolve_overlap_conflicts_greedily(spans: List[SpanInformation]) -> List[SpanInformation]:
        """
        Given a set of spans, removes spans which overlap by evaluating the difference
        in probability between one being labeled and the other explicitly having no label
        and vice-versa. The worst case time complexity of this method is `O(k * n^4)` where `n`
        is the length of the sentence that the spans were enumerated from (and therefore
        `k * m^2` complexity with respect to the number of spans `m`) and `k` is the
        number of conflicts. However, in practice, there are very few conflicts. Hopefully.

        This function modifies `spans` to remove overlapping spans.

        # Parameters

        spans : `List[SpanInformation]`, required.
            A list of spans, where each span is a `namedtuple` containing the
            following attributes:

            start : `int`
                The start index of the span.
            end : `int`
                The exclusive end index of the span.
            no_label_prob : `float`
                The probability of this span being assigned the `NO-LABEL` label.
            label_prob : `float`
                The probability of the most likely label.

        # Returns

        A modified list of `spans`, with the conflicts resolved by considering local
        differences between pairs of spans and removing one of the two spans.
        """
        conflicts_exist = True
        while conflicts_exist:
            conflicts_exist = False
            for span1_index, span1 in enumerate(spans):
                for span2_index, span2 in list(enumerate(spans))[span1_index + 1 :]:
                    if (
                        span1.start < span2.start < span1.end < span2.end
                        or span2.start < span1.start < span2.end < span1.end
                    ):
                        # The spans overlap.
                        conflicts_exist = True
                        # What's the more likely situation: that span2 was labeled
                        # and span1 was unlabled, or that span1 was labeled and span2
                        # was unlabled? In the first case, we delete span2 from the
                        # set of spans to form the tree - in the second case, we delete
                        # span1.
                        if (
                            span1.no_label_prob + span2.label_prob
                            < span2.no_label_prob + span1.label_prob
                        ):
                            spans.pop(span2_index)
                        else:
                            spans.pop(span1_index)
                        break
        return spans

    @staticmethod
    def construct_tree_from_spans(
        spans_to_labels: Dict[Tuple[int, int], str], sentence: List[str], pos_tags: List[str] = None
    ) -> Tree:
        """
        # Parameters

        spans_to_labels : `Dict[Tuple[int, int], str]`, required.
            A mapping from spans to constituency labels.
        sentence : `List[str]`, required.
            A list of tokens forming the sentence to be parsed.
        pos_tags : `List[str]`, optional (default = None)
            A list of the pos tags for the words in the sentence, if they
            were either predicted or taken as input to the model.

        # Returns

        An `nltk.Tree` constructed from the labelled spans.
        """

        def assemble_subtree(start: int, end: int):
            if (start, end) in spans_to_labels:
                # Some labels contain nested spans, e.g S-VP.
                # We actually want to create (S (VP ...)) nodes
                # for these labels, so we split them up here.
                labels: List[str] = spans_to_labels[(start, end)].split("-")
            else:
                labels = None

            # This node is a leaf.
            if end - start == 1:
                word = sentence[start]
                pos_tag = pos_tags[start] if pos_tags is not None else "XX"
                tree = Tree(pos_tag, [word])
                if labels is not None and pos_tags is not None:
                    # If POS tags were passed explicitly,
                    # they are added as pre-terminal nodes.
                    while labels:
                        tree = Tree(labels.pop(), [tree])
                elif labels is not None:
                    # Otherwise, we didn't want POS tags
                    # at all.
                    tree = Tree(labels.pop(), [word])
                    while labels:
                        tree = Tree(labels.pop(), [tree])
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
            if labels is not None:
                while labels:
                    children = [Tree(labels.pop(), children)]
            return children

        tree = assemble_subtree(0, len(sentence))
        return tree[0]

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        all_metrics["tag_accuracy"] = self.tag_accuracy.get_metric(reset=reset)
        if self._evalb_score is not None:
            evalb_metrics = self._evalb_score.get_metric(reset=reset)
            all_metrics.update(evalb_metrics)
        return all_metrics

from typing import Dict, Tuple, List, Optional, NamedTuple, Any
from overrides import overrides

import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import last_dim_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import EvalbBracketingScorer
from allennlp.common.checks import ConfigurationError
import numpy
from sortedcontainers import SortedList

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
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
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
                 pos_tag_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 evalb_directory_path: str = None) -> None:
        super(SpanConstituencyParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.span_extractor = span_extractor
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.feedforward_layer = TimeDistributed(feedforward_layer) if feedforward_layer else None
        self.pos_tag_embedding = pos_tag_embedding or None
        if feedforward_layer is not None:
            output_dim = feedforward_layer.get_output_dim()
        else:
            output_dim = span_extractor.get_output_dim()

        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_classes))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()
        check_dimensions_match(representation_dim,
                               encoder.get_input_dim(),
                               "representation dim (tokens + optional POS tags)",
                               "encoder input dim")
        check_dimensions_match(encoder.get_output_dim(),
                               span_extractor.get_input_dim(),
                               "encoder input dim",
                               "span extractor input dim")
        if feedforward_layer is not None:
            check_dimensions_match(span_extractor.get_output_dim(),
                                   feedforward_layer.get_input_dim(),
                                   "span extractor output dim",
                                   "feedforward input dim")

        self.tag_accuracy = CategoricalAccuracy()

        if evalb_directory_path is not None:
            self._evalb_score = EvalbBracketingScorer(evalb_directory_path)
        else:
            self._evalb_score = None
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                spans: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                pos_tags: Dict[str, torch.LongTensor] = None,
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
        metadata : List[Dict[str, Any]], required.
            A dictionary of metadata for each batch element which has keys:
                tokens : ``List[str]``, required.
                    The original string tokens in the sentence.
                gold_tree : ``nltk.Tree``, optional (default = None)
                    Gold NLTK trees for use in evaluation.
                pos_tags : ``List[str]``, optional.
                    The POS tags for the sentence. These can be used in the
                    model as embedded features, but they are passed here
                    in addition for use in constructing the tree.
        pos_tags : ``torch.LongTensor``, optional (default = None)
            The output of a ``SequenceLabelField`` containing POS tags.
        span_labels : ``torch.LongTensor``, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape ``(batch_size, num_spans)``.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        spans : ``torch.LongTensor``
            The original spans tensor.
        tokens : ``List[List[str]]``, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : ``List[List[str]]``, required.
            A list of POS tags in the sentence for each element in the batch.
        num_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in ``enumerated_spans``.
        loss : ``torch.FloatTensor``, optional
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
        class_probabilities = last_dim_softmax(logits, span_mask.unsqueeze(-1))

        output_dict = {
                "class_probabilities": class_probabilities,
                "spans": spans,
                "tokens": [meta["tokens"] for meta in metadata],
                "pos_tags": [meta.get("pos_tags") for meta in metadata],
                "num_spans": num_spans
        }
        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
            self.tag_accuracy(class_probabilities, span_labels, span_mask)
            output_dict["loss"] = loss

        # The evalb score is expensive to compute, so we only compute
        # it for the validation and test sets.
        batch_gold_trees = [meta.get("gold_tree") for meta in metadata]
        if all(batch_gold_trees) and self._evalb_score is not None and not self.training:
            gold_pos_tags: List[List[str]] = [list(zip(*tree.pos()))[1]
                                              for tree in batch_gold_trees]
            predicted_top1_trees = self.construct_topk_trees(class_probabilities.cpu().data,
                                                   spans.cpu().data,
                                                   num_spans.data,
                                                   output_dict["tokens"],
                                                   gold_pos_tags,
                                                        num_trees=1)
            self._evalb_score([trees[0] for tree in predicted_top1_trees], batch_gold_trees)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constructs an NLTK ``Tree`` given the scored spans. We also switch to exclusive
        span ends when constructing the tree representation, because it makes indexing
        into lists cleaner for ranges of text, rather than individual indices.

        Finally, for batch prediction, we will have padded spans and class probabilities.
        In order to make this less confusing, we remove all the padded spans and
        distributions from ``spans`` and ``class_probabilities`` respectively.
        """
        all_predictions = output_dict['class_probabilities'].cpu().data
        all_spans = output_dict["spans"].cpu().data

        all_sentences = output_dict["tokens"]
        all_pos_tags = output_dict["pos_tags"] if all(output_dict["pos_tags"]) else None
        num_spans = output_dict["num_spans"].data

        batch_size = all_predictions.size(0)
        output_dict["spans"] = [all_spans[i, :num_spans[i]] for i in range(batch_size)]
        output_dict["class_probabilities"] = [all_predictions[i, :num_spans[i], :] for i in range(batch_size)]

        top_k_trees = self.construct_topk_trees(num_trees=16,
                                                predictions=all_predictions,
                                                all_spans=all_spans,
                                                num_spans=num_spans,
                                                sentences=all_sentences,
                                                pos_tags=all_pos_tags)

        output_dict["trees"] = [trees[0] for trees in top_k_trees]
        output_dict['top_k_trees'] = top_k_trees
        return output_dict

    def compute_k_best(self,
                       sentence: List[str],
                       pos_tags: List[str],
                       label_probabilities: torch.FloatTensor,
                       span_to_index: Dict[Tuple[int, int], int],
                       num_trees: List[int],
                       distinguish_between_labels: bool = False):
        """
        :param sentence: The sentence for which top-k parses are being computed.
        :param pos_tags: Part of speech tags for every token in the input sentence.
        :param label_probabilities: A numpy array of shape (num_spans, num_labels)
        :param span_to_index: A dictionary mapping span indices to column indices in
        label_log_probabilities.
        :param no_label_id: The id of the empty label.
        :param num_trees: The number of parses required.
        :param distinguish_between_labels: Whether to distinguish between different labels for the
        top k trees.
        :return: A list of the num_tree parses, and their log probabilities.
        """
        empty_label_index = self.vocab.get_token_index("NO-LABEL", "labels")
        all_labels = [self.vocab.get_token_from_index(index, "labels").split("-") for index in
                  range(self.vocab.get_vocab_size("labels"))]
        label_probabilities_np = label_probabilities.cpu().numpy()
        if not distinguish_between_labels:
            temp = numpy.zeros((len(span_to_index), 2))
            temp[:, 0] = label_probabilities_np[:, empty_label_index]
            temp[:, 1] = 1 - label_probabilities_np[:, empty_label_index]

            span_to_label = {}
            # To ensure that the empty label does not have the maximum probability for any span.
            label_probabilities_np[:, empty_label_index] = -1
            for span, span_index in span_to_index.items():
                label_index = label_probabilities_np[span_index, :].argmax()
                span_to_label[span] = all_labels[label_index]

            label_probabilities_np = temp
            empty_label_index = 0


        label_log_probabilities_np = numpy.log(label_probabilities_np + 10 ** -5)
        correction_term = numpy.sum(label_log_probabilities_np[:, empty_label_index])
        label_log_probabilities_np -= label_log_probabilities_np[:, empty_label_index]\
            .reshape((len(span_to_index), 1))
        cache = {}

        def helper(left, right, must_be_constituent):
            assert left < right
            span = (left, right)
            if span in cache:
                return cache[span]


            if not distinguish_between_labels:
                labels = [(), span_to_label[span]]
            else:
                labels = all_labels

            span_index = span_to_index[span]
            actions = list(enumerate(label_log_probabilities_np[span_index, :]))
            actions.sort(key=lambda x: - x[1])
            actions = actions[:num_trees]

            if right - left == 1:
                word = sentence[left]
                pos_tag = pos_tags[left]
                options = []
                for label_index, score in actions:
                    tree = Tree(pos_tag, [word])
                    if label_index != empty_label_index:
                        label = list(labels[label_index])
                        while label:
                            tree = Tree(label.pop(), [tree])
                    options.append(([tree], score))
                cache[span] = options
            else:
                children_options = SortedList(key=lambda x: - x[1])
                for split in range(left + 1, right):
                    left_trees_options = helper(left, split, must_be_constituent=True)
                    right_trees_options = helper(split, right, must_be_constituent=False)
                    for (left_trees, left_score) in left_trees_options:
                        assert len(left_trees) == 1, 'Toa avoid duplicates we require that left' \
                                                     'trees are constituents.'
                        for (right_trees, right_score) in right_trees_options:
                            children = left_trees + right_trees
                            score = left_score + right_score
                            if len(children_options) < num_trees:
                                children_options.add((children, score))
                            elif children_options[-1][1] < score:
                                del children_options[-1]
                                children_options.add((children, score))

                options = SortedList(key=lambda x: - x[1])
                for (label_index, action_score) in actions:
                    for (children, children_score) in children_options:
                        option_score = action_score + children_score
                        if label_index != empty_label_index:
                            label = list(labels[label_index])
                            while label:
                                children = [Tree(label.pop(), children)]
                            option = children
                        elif must_be_constituent:
                            continue
                        else:
                            option = children
                        if len(options) < num_trees:
                            options.add((option, option_score))
                        elif options[-1][1] < option_score:
                            del options[-1]
                            options.add((option, option_score))
                        else:
                            break
                cache[span] = options
            return cache[span]

        trees_and_scores = helper(0, len(sentence), must_be_constituent=True)[:num_trees]
        trees = []
        scores = []
        for tree, score in trees_and_scores:
            assert len(tree) == 1
            trees.append(tree[0])
            scores.append(score + correction_term)
        return trees, scores

    def construct_topk_trees(self,
                        predictions: torch.FloatTensor,
                        all_spans: torch.LongTensor,
                        num_spans: torch.LongTensor,
                        sentences: List[List[str]],
                        pos_tags: List[List[str]],
                             num_trees: int,) -> List[Tree]:
        """
        Construct ``nltk.Tree``'s for each batch element by greedily nesting spans.
        The trees use exclusive end indices, which contrasts with how spans are
        represented in the rest of the model.

        Parameters
        ----------
        predictions : ``torch.FloatTensor``, required.
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        all_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the span
            indices we scored.
        num_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in ``enumerated_spans``.
        sentences : ``List[List[str]]``, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : ``List[List[str]]``, optional (default = None).
            A list of POS tags for each word in the sentence for each element
            in the batch.
        num_trees: ``int``, required.
            The number of trees to be returned for each sentence.

        Returns
        -------
        A ``List[Tree]`` containing the decoded trees for each element in the batch.
        :param sentences:
        :param predictions:
        :param all_spans:
        :param num_trees:
        """
        # Switch to using exclusive end spans.
        exclusive_end_spans = all_spans.clone().cpu().numpy()
        exclusive_end_spans[:, :, -1] += 1

        trees: List[List[Tree]] = []
        for batch_index in range(len(sentences)):
            span_to_index = {}
            for span_index in range(num_spans[batch_index]):
                span = (exclusive_end_spans[batch_index, span_index, 0],
                        exclusive_end_spans[batch_index, span_index, 1])
                span_to_index[span] = span_index
            top_k_trees, _ = self.compute_k_best(sentences[batch_index],
                                                         pos_tags[batch_index],
                                                         predictions[batch_index, :, :],
                                                         span_to_index,
                                                         num_trees=num_trees)
            trees.append(top_k_trees)
        return trees

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        all_metrics["tag_accuracy"] = self.tag_accuracy.get_metric(reset=reset)
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
        pos_tag_embedding_params = params.pop("pos_tag_embedding", None)
        if pos_tag_embedding_params is not None:
            pos_tag_embedding = Embedding.from_params(vocab, pos_tag_embedding_params)
        else:
            pos_tag_embedding = None
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        evalb_directory_path = params.pop("evalb_directory_path", None)
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   span_extractor=span_extractor,
                   encoder=encoder,
                   feedforward_layer=feedforward_layer,
                   pos_tag_embedding=pos_tag_embedding,
                   initializer=initializer,
                   regularizer=regularizer,
                   evalb_directory_path=evalb_directory_path)

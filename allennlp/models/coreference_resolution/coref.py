import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, GatedSum
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores

logger = logging.getLogger(__name__)


@Model.register("coref")
class CoreferenceResolver(Model):
    """
    This `Model` implements the coreference resolution model described in
    [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/pdf/1804.05392.pdf)
    by Lee et al., 2018.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder`
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : `FeedForward`
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward : `FeedForward`
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size : `int`
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width : `int`
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    coarse_to_fine: bool, optional (default = False)
        Whether or not to apply the coarse-to-fine filtering.
    inference_order: int, optional (default = 1)
        The number of inference orders. When greater than 1, the span representations are
        updated and coreference scores re-computed.
    lexical_dropout : `int`
        The probability of dropping out dimensions of the embedded text.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        mention_feedforward: FeedForward,
        antecedent_feedforward: FeedForward,
        feature_size: int,
        max_span_width: int,
        spans_per_word: float,
        max_antecedents: int,
        coarse_to_fine: bool = False,
        inference_order: int = 1,
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._mention_scorer = TimeDistributed(
            torch.nn.Linear(mention_feedforward.get_output_dim(), 1)
        )
        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._antecedent_scorer = TimeDistributed(
            torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1)
        )

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=text_field_embedder.get_output_dim()
        )

        # 10 possible distance buckets.
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(
            embedding_dim=feature_size, num_embeddings=self._num_distance_buckets
        )

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._coarse_to_fine = coarse_to_fine
        if self._coarse_to_fine:
            self._coarse2fine_scorer = torch.nn.Linear(
                mention_feedforward.get_input_dim(), mention_feedforward.get_input_dim()
            )
        self._inference_order = inference_order
        if self._inference_order > 1:
            self._span_updating_gated_sum = GatedSum(mention_feedforward.get_input_dim())

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text: TextFieldTensors,
        spans: torch.IntTensor,
        span_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        text : `TextFieldTensors`, required.
            The output of a `TextField` representing the text of
            the document.
        spans : `torch.IntTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a `ListField[SpanField]` of
            indices into the text of the document.
        span_labels : `torch.IntTensor`, optional (default = None).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : `List[Dict[str, Any]]`, optional (default = None).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.

        # Returns

        An output dictionary consisting of:
        top_spans : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep, 2)` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : `torch.IntTensor`
            A tensor of shape `(num_spans_to_keep, max_antecedents)` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep)` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        batch_size = spans.size(0)
        document_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text)

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1)
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))
        num_spans_to_keep = min(num_spans_to_keep, num_spans)

        # Shape: (batch_size, num_spans)
        span_mention_scores = self._mention_scorer(
            self._mention_feedforward(span_embeddings)
        ).squeeze(-1)
        # Shape: (batch_size, num_spans) for all 3 tensors
        top_span_mention_scores, top_span_mask, top_span_indices = util.masked_topk(
            span_mention_scores, span_mask, num_spans_to_keep
        )

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)
        # Shape: (batch_size, num_spans_to_keep, embedding_size)
        top_span_embeddings = util.batched_index_select(
            span_embeddings, top_span_indices, flat_top_span_indices
        )

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Now that we have our variables in terms of num_spans_to_keep, we need to
        # compare span pairs to decide each span's antecedent. Each span can only
        # have prior spans as antecedents, and we only consider up to max_antecedents
        # prior spans. So the first thing we do is construct a matrix mapping a span's
        # index to the indices of its allowed antecedents.

        # Once we have this matrix, we reformat our variables again to get embeddings
        # for all valid antecedents for each span. This gives us variables with shapes
        # like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
        # we can use to make coreference decisions between valid span pairs.

        if self._coarse_to_fine:
            pruned_antecedents = self._coarse_to_fine_pruning(
                top_span_embeddings, top_span_mention_scores, top_span_mask, max_antecedents
            )
        else:
            pruned_antecedents = self._distance_pruning(
                top_span_embeddings, top_span_mention_scores, max_antecedents
            )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents) for all 4 tensors
        (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        ) = pruned_antecedents

        flat_top_antecedent_indices = util.flatten_and_batch_shift_indices(
            top_antecedent_indices, num_spans_to_keep
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        top_antecedent_embeddings = util.batched_index_select(
            top_span_embeddings, top_antecedent_indices, flat_top_antecedent_indices
        )
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(
            top_span_embeddings,
            top_antecedent_embeddings,
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
        )

        for _ in range(self._inference_order - 1):
            dummy_mask = top_antecedent_mask.new_ones(batch_size, num_spans_to_keep, 1)
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents,)
            top_antecedent_with_dummy_mask = torch.cat([dummy_mask, top_antecedent_mask], -1)
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
            attention_weight = util.masked_softmax(
                coreference_scores, top_antecedent_with_dummy_mask, memory_efficient=True
            )
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents, embedding_size)
            top_antecedent_with_dummy_embeddings = torch.cat(
                [top_span_embeddings.unsqueeze(2), top_antecedent_embeddings], 2
            )
            # Shape: (batch_size, num_spans_to_keep, embedding_size)
            attended_embeddings = util.weighted_sum(
                top_antecedent_with_dummy_embeddings, attention_weight
            )
            # Shape: (batch_size, num_spans_to_keep, embedding_size)
            top_span_embeddings = self._span_updating_gated_sum(
                top_span_embeddings, attended_embeddings
            )

            # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
            top_antecedent_embeddings = util.batched_index_select(
                top_span_embeddings, top_antecedent_indices, flat_top_antecedent_indices
            )
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
            coreference_scores = self._compute_coreference_scores(
                top_span_embeddings,
                top_antecedent_embeddings,
                top_partial_coreference_scores,
                top_antecedent_mask,
                top_antecedent_offsets,
            )

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1

        output_dict = {
            "top_spans": top_spans,
            "antecedent_indices": top_antecedent_indices,
            "predicted_antecedents": predicted_antecedents,
        }
        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            # Shape: (batch_size, num_spans_to_keep, 1)
            pruned_gold_labels = util.batched_index_select(
                span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices
            )

            # Shape: (batch_size, num_spans_to_keep, max_antecedents)
            antecedent_labels = util.batched_index_select(
                pruned_gold_labels, top_antecedent_indices, flat_top_antecedent_indices
            ).squeeze(-1)
            antecedent_labels = util.replace_masked_values(
                antecedent_labels, top_antecedent_mask, -100
            )

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(
                pruned_gold_labels, antecedent_labels
            )
            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.
            coreference_log_probs = util.masked_log_softmax(
                coreference_scores, top_span_mask.unsqueeze(-1)
            )
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(
                top_spans, top_antecedent_indices, predicted_antecedents, metadata
            )

            output_dict["loss"] = negative_marginal_log_likelihood

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        # Parameters

        output_dict : `Dict[str, torch.Tensor]`, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        # Returns

        The same output dictionary, but with an additional `clusters` key:

        clusters : `List[List[List[Tuple[int, int]]]]`
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into `antecedent_indices` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of `batch_top_spans`
        # for each antecedent we considered.
        batch_antecedent_indices = output_dict["antecedent_indices"].detach().cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, predicted_antecedents, antecedent_indices in zip(
            batch_top_spans, batch_predicted_antecedents, batch_antecedent_indices
        ):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                # To do this, we find the row in `antecedent_indices`
                # corresponding to this span we are considering.
                # The predicted antecedent is then an index into this list
                # of indices, denoting the span from `top_spans` which is the
                # most likely antecedent.
                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (
                    top_spans[predicted_index, 0].item(),
                    top_spans[predicted_index, 1].item(),
                )

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span[0].item(), span[1].item()
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {
            "coref_precision": coref_precision,
            "coref_recall": coref_recall,
            "coref_f1": coref_f1,
            "mention_recall": mention_recall,
        }

    @staticmethod
    def _generate_valid_antecedents(
        num_spans_to_keep: int, max_antecedents: int, device: int
    ) -> Tuple[torch.IntTensor, torch.IntTensor, torch.BoolTensor]:
        """
        This method generates possible antecedents per span which survived the pruning
        stage. This procedure is `generic across the batch`. The reason this is the case is
        that each span in a batch can be coreferent with any previous span, but here we
        are computing the possible `indices` of these spans. So, regardless of the batch,
        the 1st span _cannot_ have any antecedents, because there are none to select from.
        Similarly, each element can only predict previous spans, so this returns a matrix
        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
        (i - 1) - j if j <= i, or zero otherwise.

        # Parameters

        num_spans_to_keep : `int`, required.
            The number of spans that were kept while pruning.
        max_antecedents : `int`, required.
            The maximum number of antecedent spans to consider for every span.
        device : `int`, required.
            The CUDA device to use.

        # Returns

        valid_antecedent_indices : `torch.LongTensor`
            The indices of every antecedent to consider with respect to the top k spans.
            Has shape `(num_spans_to_keep, max_antecedents)`.
        valid_antecedent_offsets : `torch.LongTensor`
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            Has shape `(1, max_antecedents)`.
        valid_antecedent_mask : `torch.BoolTensor`
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            Has shape `(1, num_spans_to_keep, max_antecedents)`.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction.
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets

        # In our matrix of indices, the upper triangular part will be negative
        # because the offsets will be > the target indices. We want to mask these,
        # because these are exactly the indices which we don't want to predict, per span.
        # Shape: (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_mask = (raw_antecedent_indices >= 0).unsqueeze(0)

        # Shape: (num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_mask

    def _distance_pruning(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        max_antecedents: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generates antecedents for each span and prunes down to `max_antecedents`. This method
        prunes antecedents only based on distance (i.e. number of intervening spans). The closest
        antecedents are kept.

        # Parameters

        top_span_embeddings: torch.FloatTensor, required.
            The embeddings of the top spans.
            (batch_size, num_spans_to_keep, embedding_size).
        top_span_mention_scores: torch.FloatTensor, required.
            The mention scores of the top spans.
            (batch_size, num_spans_to_keep).
        max_antecedents: int, required.
            The maximum number of antecedents to keep for each span.

        # Returns

        top_partial_coreference_scores: torch.FloatTensor
            The partial antecedent scores for each span-antecedent pair. Computed by summing
            the span mentions scores of the span and the antecedent. This score is partial because
            compared to the full coreference scores, it lacks the interaction term
            w * FFNN([g_i, g_j, g_i * g_j, features]).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mask: torch.BoolTensor
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_offsets: torch.LongTensor
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_indices: torch.LongTensor
            The indices of every antecedent to consider with respect to the top k spans.
            (batch_size, num_spans_to_keep, max_antecedents)
        """
        # These antecedent matrices are independent of the batch dimension - they're just a function
        # of the span's position in top_spans.
        # The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        num_spans_to_keep = top_span_embeddings.size(1)
        device = util.get_device_of(top_span_embeddings)

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        (
            top_antecedent_indices,
            top_antecedent_offsets,
            top_antecedent_mask,
        ) = self._generate_valid_antecedents(  # noqa
            num_spans_to_keep, max_antecedents, device
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mention_scores = util.flattened_index_select(
            top_span_mention_scores.unsqueeze(-1), top_antecedent_indices
        ).squeeze(-1)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents) * 4
        top_partial_coreference_scores = (
            top_span_mention_scores.unsqueeze(-1) + top_antecedent_mention_scores
        )
        top_antecedent_indices = top_antecedent_indices.unsqueeze(0).expand_as(
            top_partial_coreference_scores
        )
        top_antecedent_offsets = top_antecedent_offsets.unsqueeze(0).expand_as(
            top_partial_coreference_scores
        )
        top_antecedent_mask = top_antecedent_mask.expand_as(top_partial_coreference_scores)

        return (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        )

    def _coarse_to_fine_pruning(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        top_span_mask: torch.BoolTensor,
        max_antecedents: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generates antecedents for each span and prunes down to `max_antecedents`. This method
        prunes antecedents using a fast bilinar interaction score between a span and a candidate
        antecedent, and the highest-scoring antecedents are kept.

        # Parameters

        top_span_embeddings: torch.FloatTensor, required.
            The embeddings of the top spans.
            (batch_size, num_spans_to_keep, embedding_size).
        top_span_mention_scores: torch.FloatTensor, required.
            The mention scores of the top spans.
            (batch_size, num_spans_to_keep).
        top_span_mask: torch.BoolTensor, required.
            The mask for the top spans.
            (batch_size, num_spans_to_keep).
        max_antecedents: int, required.
            The maximum number of antecedents to keep for each span.

        # Returns

        top_partial_coreference_scores: torch.FloatTensor
            The partial antecedent scores for each span-antecedent pair. Computed by summing
            the span mentions scores of the span and the antecedent as well as a bilinear
            interaction term. This score is partial because compared to the full coreference scores,
            it lacks the interaction term
            w * FFNN([g_i, g_j, g_i * g_j, features]).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mask: torch.BoolTensor
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_offsets: torch.LongTensor
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_indices: torch.LongTensor
            The indices of every antecedent to consider with respect to the top k spans.
            (batch_size, num_spans_to_keep, max_antecedents)
        """
        batch_size, num_spans_to_keep = top_span_embeddings.size()[:2]
        device = util.get_device_of(top_span_embeddings)

        # Shape: (1, num_spans_to_keep, num_spans_to_keep)
        _, _, valid_antecedent_mask = self._generate_valid_antecedents(
            num_spans_to_keep, num_spans_to_keep, device
        )

        mention_one_score = top_span_mention_scores.unsqueeze(1)
        mention_two_score = top_span_mention_scores.unsqueeze(2)
        bilinear_weights = self._coarse2fine_scorer(top_span_embeddings).transpose(1, 2)
        bilinear_score = torch.matmul(top_span_embeddings, bilinear_weights)
        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep); broadcast op
        partial_antecedent_scores = mention_one_score + mention_two_score + bilinear_score

        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep); broadcast op
        span_pair_mask = top_span_mask.unsqueeze(-1) & valid_antecedent_mask

        # Shape:
        # (batch_size, num_spans_to_keep, max_antecedents) * 3
        (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_indices,
        ) = util.masked_topk(partial_antecedent_scores, span_pair_mask, max_antecedents)

        top_span_range = util.get_range_vector(num_spans_to_keep, device)
        # Shape: (num_spans_to_keep, num_spans_to_keep); broadcast op
        valid_antecedent_offsets = top_span_range.unsqueeze(-1) - top_span_range.unsqueeze(0)

        # TODO: we need to make `batched_index_select` more general to make this less awkward.
        top_antecedent_offsets = util.batched_index_select(
            valid_antecedent_offsets.unsqueeze(0)
            .expand(batch_size, num_spans_to_keep, num_spans_to_keep)
            .reshape(batch_size * num_spans_to_keep, num_spans_to_keep, 1),
            top_antecedent_indices.view(-1, max_antecedents),
        ).reshape(batch_size, num_spans_to_keep, max_antecedents)

        return (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        )

    def _compute_span_pair_embeddings(
        self,
        top_span_embeddings: torch.FloatTensor,
        antecedent_embeddings: torch.FloatTensor,
        antecedent_offsets: torch.FloatTensor,
    ):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : `torch.IntTensor`, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        span_pair_embeddings : `torch.FloatTensor`
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat(
            [
                target_embeddings,
                antecedent_embeddings,
                antecedent_embeddings * target_embeddings,
                antecedent_distance_embeddings,
            ],
            -1,
        )
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(
        top_span_labels: torch.IntTensor, antecedent_labels: torch.IntTensor
    ):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        # Parameters

        top_span_labels : `torch.IntTensor`, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : `torch.IntTensor`, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        pairwise_labels_with_dummy_label : `torch.FloatTensor`
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_antecedent_embeddings: torch.FloatTensor,
        top_partial_coreference_scores: torch.FloatTensor,
        top_antecedent_mask: torch.BoolTensor,
        top_antecedent_offsets: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the kept spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size)
        top_antecedent_embeddings: `torch.FloatTensor`, required.
            The embeddings of antecedents for each span candidate. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        top_partial_coreference_scores : `torch.FloatTensor`, required.
            Sum of span mention score and antecedent mention score. The coarse to fine settings
            has an additional term which is the coarse bilinear score.
            (batch_size, num_spans_to_keep, max_antecedents).
        top_antecedent_mask : `torch.BoolTensor`, required.
            The mask for valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents).
        top_antecedent_offsets : `torch.FloatTensor`, required.
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        coreference_scores : `torch.FloatTensor`
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(
            top_span_embeddings, top_antecedent_embeddings, top_antecedent_offsets
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
            self._antecedent_feedforward(span_pair_embeddings)
        ).squeeze(-1)
        antecedent_scores += top_partial_coreference_scores
        antecedent_scores = util.replace_masked_values(
            antecedent_scores, top_antecedent_mask, -1e20
        )

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = antecedent_scores.new_zeros(*shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores

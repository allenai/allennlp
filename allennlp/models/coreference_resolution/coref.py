import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("coref")
class CoreferenceResolver(Model):
    """
    This ``Model`` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representation are scored and use to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 mention_feedforward: FeedForward,
                 antecedent_feedforward: FeedForward,
                 feature_size: int,
                 max_span_width: int,
                 spans_per_word: float,
                 max_antecedents: int,
                 lexical_dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(CoreferenceResolver, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._mention_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1))
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))
        self._head_scorer = TimeDistributed(torch.nn.Linear(context_layer.get_output_dim(), 1))

        # 10 possible distance buckets.
        self._distance_embedding = Embedding(10, feature_size)
        self._span_width_embedding = Embedding(max_span_width, feature_size)

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    def _compute_head_attention(self,
                                head_scores: torch.FloatTensor,
                                text_embeddings: torch.FloatTensor,
                                span_ends: torch.IntTensor,
                                span_size: torch.IntTensor):
        """
        Parameters
        ----------
        head_scores : torch.FloatTensor
            Unnormalized attention scores for every word. This score is shared for every candidate. The
            only way in which the attention weights differ over different spans is in the set of words
            over which they are normalized.
        text_embeddings: torch.FloatTensor
            The embeddings over which we are computing a weighted sum.
        span_ends: torch.IntTensor
            The end indices of all the span candidates.
        span_size : torch.IntTensor
            The size each span candidate.
        Returns
        -------
        attended_text_embeddings : torch.FloatTensor
            The result of applying attention over all words within each candidate span.
        """
        # Shape: (1, 1, max_span_width)
        head_offsets = util.get_range_vector(self._max_span_width, text_embeddings.is_cuda).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_span_width)
        head_mask = (head_offsets <= span_size).float()
        raw_head_indices = span_ends - head_offsets
        head_mask = head_mask * (raw_head_indices >= 0).float()
        head_indices = F.relu(raw_head_indices.float()).long()

        # Shape: (batch_size * num_spans * max_span_width)
        flat_head_indices = util.flatten_and_batch_shift_indices(head_indices, text_embeddings.size(1))

        # Shape: (batch_size, num_spans, max_span_width, embedding_size)
        span_text_embeddings = util.batched_index_select(text_embeddings, head_indices, flat_head_indices)

        # Shape: (batch_size, num_spans, max_span_width)
        span_head_scores = util.batched_index_select(head_scores, head_indices, flat_head_indices).squeeze(-1)
        span_head_scores += head_mask.float().log()

        # Shape: (batch_size * num_spans, max_span_width)
        flat_span_head_scores = span_head_scores.view(-1, self._max_span_width)
        flat_span_head_weights = F.softmax(flat_span_head_scores)

        # Shape: (batch_size * num_spans, 1, max_span_width)
        flat_span_head_weights = flat_span_head_weights.unsqueeze(1)

        # Shape: (batch_size * num_spans, max_span_width, embedding_size)
        flat_span_text_embeddings = span_text_embeddings.view(-1,
                                                              self._max_span_width,
                                                              span_text_embeddings.size(-1))

        # Shape: (batch_size * num_spans, 1, embedding_size)
        flat_attended_text_embeddings = flat_span_head_weights.bmm(flat_span_text_embeddings)

        # Shape: (batch_size, num_spans, embedding_size)
        attended_text_embeddings = flat_attended_text_embeddings.view(text_embeddings.size(0),
                                                                      span_ends.size(1), -1)
        return attended_text_embeddings

    def _compute_span_representations(self,
                                      text_embeddings: torch.FloatTensor,
                                      text_mask: torch.FloatTensor,
                                      span_starts: torch.IntTensor,
                                      span_ends: torch.IntTensor):
        """
        Computes an embedded representation of every candidate span. This is a concatenation
        of the contextualized endpoints of the span, an embedded representation of the width of
        the span and the result of applying attention to the words in the span.

        Parameters
        ----------
        text_embeddings: torch.FloatTensor
            The embeddings over which we are computing a weighted sum.
        text_mask: torch.FloatTensor
            Mask representing non-padding entries of ``text_embeddings``.
        span_starts : torch.IntTensor
            The start of each span candidate.
        span_ends : torch.IntTensor
            The end of each span candidate.
        Returns
        -------
        span_embeddings : torch.FloatTensor
            An embedded representation of every candidate span with shape:
            (batch_size, num_spans, context_layer.get_output_dim() * 2 + embedding_size + feature_size)
        """
        # Shape: (batch_size, text_length, embedding_size)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        # Shape: (batch_size, num_spans, embedding_size)
        start_embeddings = util.batched_index_select(contextualized_embeddings, span_starts.squeeze(-1))
        end_embeddings = util.batched_index_select(contextualized_embeddings, span_ends.squeeze(-1))

        # Compute and embed the span_width (strictly speaking the span_width - 1)
        # Shape: (batch_size, num_spans, 1)
        span_width = span_ends - span_starts
        # Shape: (batch_size, num_spans, embedding_size)
        span_width_embeddings = self._span_width_embedding(span_width.squeeze(-1))

        # Shape: (batch_size, text_len, 1)
        head_scores = self._head_scorer(contextualized_embeddings)

        # Shape: (batch_size, num_spans, embedding_size)
        attended_text_embeddings = self._compute_head_attention(head_scores,
                                                                text_embeddings,
                                                                span_ends,
                                                                span_width)
        # (batch_size, num_spans, context_layer.get_output_dim() * 2 + embedding_size + feature_size)
        span_embeddings = torch.cat([start_embeddings,
                                     end_embeddings,
                                     span_width_embeddings,
                                     attended_text_embeddings], -1)
        return span_embeddings

    @staticmethod
    def _prune_and_sort_spans(mention_scores: torch.FloatTensor, num_spans_to_keep: int):
        """
        Parameters
        ----------
        mention_scores: torch.FloatTensor
            The mention score for every candidate
        num_spans_to_keep: int
            The number of spans to keep when pruning.
        Returns
        -------
        top_span_indices : torch.IntTensor
            The top-k spans according to the given mention scores. The output spans
            appear in the same order as the input spans.
        """
        # Shape: (batch_size, num_spans_to_keep, 1)
        _, top_span_indices = mention_scores.topk(num_spans_to_keep, 1)
        top_span_indices, _ = torch.sort(top_span_indices, 1)

        # Shape: (batch_size, num_spans_to_keep)
        top_span_indices = top_span_indices.squeeze(-1)
        return top_span_indices

    @staticmethod
    def _generate_antecedents(num_spans_to_keep: int, max_antecedents: int, is_cuda: bool):
        """
        Parameters
        ----------
        num_spans_to_keep: int
            The number of spans that were kept while pruning.
        max_antecedents: int
            The maximum number of antecedent spans to consider for every span.
        is_cuda: bool
            Whether the computation is being done on the GPU or not.
        Returns
        -------
        antecedent_indices : torch.IntTensor
            The indices of every antecedent to consider with respect to the top k spans.
        antecedent_offsets: torch.IntTensor
            The distance between the span and each of its antecedents.
        antecedent_log_mask: torch.FloatTensor
            The logged mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, is_cuda).unsqueeze(1)

        # Shape: (1, max_antecedents)
        antecedent_offsets = (util.get_range_vector(max_antecedents, is_cuda) + 1).unsqueeze(0)

        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - antecedent_offsets

        # Shape: (1, num_spans_to_keep, max_antecedents)
        antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

        # Shape: (num_spans_to_keep, max_antecedents)
        antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return antecedent_indices, antecedent_offsets, antecedent_log_mask

    def _compute_pairwise_inputs(self,
                                 top_span_embeddings: torch.FloatTensor,
                                 antecedent_embeddings: torch.FloatTensor,
                                 antecedent_offsets: torch.FloatTensor):
        """
        Parameters
        ----------
        top_span_embeddings: torch.FloatTensor
            Embedding representations of the top spans.
        antecedent_embeddings: torch.FloatTensor
            Embedding representaitons of the antecedent spans for each top span.
        antecedent_offsets: torch.IntTensor
            The offsets between each top span and its antecedent spans.
        Returns
        -------
        pairwise_embeddings : torch.FloatTensor
            Embedding representation of the pair of spans to consider. This includes both the original span
            representations, the elementwise similarity of the span representations, and an embedding
            representation of the distance between two spans. This is used as input to pairwise classification.
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)
        similarity_embeddings = antecedent_embeddings * target_embeddings

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(util.bucket_values(antecedent_offsets))

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0),
                                              antecedent_embeddings.size(1),
                                              antecedent_embeddings.size(2),
                                              antecedent_distance_embeddings.size(-1))
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        pairwise_embeddings = torch.cat([target_embeddings,
                                         antecedent_embeddings,
                                         similarity_embeddings,
                                         antecedent_distance_embeddings], -1)
        return pairwise_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(top_span_labels: torch.IntTensor,
                                        antecedent_labels: torch.IntTensor):
        """
        Parameters
        ----------
        top_span_labels: torch.IntTensor
            The cluster id label for every span.
        antecedent_labels: torch.IntTensor
            The cluster id label for every antecedent span.
        Returns
        -------
        augmented_label: torch.FloatTensor
            A binary indicator for every pair of spans. This label is one if and only if the pair of spans belong
            to the same cluster. The labels are augmented with a dummy antecedent at the zeroth position, which
            represents the prediction that a span does not have any antecedent.
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        augmented_labels = torch.cat([dummy_labels, pairwise_labels], -1)
        return augmented_labels

    @staticmethod
    def _compute_negative_marginal_log_likelihood(augmented_antecedent_scores: torch.FloatTensor,
                                                  augmented_labels: torch.IntTensor,
                                                  top_span_mask: torch.FloatTensor):
        """
        Parameters
        ----------
        augmented_antecedent_scores: torch.FloatTensor
            The pairwise between every span and its possible antecedents.
        augmented_labels: torch.IntTensor
            A binary indicator for whether it is consistent with the data for a span to choose an antecedent.
        Returns
        -------
        negative_marginal_log_likelihood: torch.FloatTensor
            The negative marginal loglikelihood of the gold cluster labels. This computes the log of the sum of the
            probabilities of all antecedent predictions that would be consistent with the data. The computation is
            performed in log-space for numerical stability.
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        gold_scores = augmented_antecedent_scores + augmented_labels.log()

        # Shape: (batch_size, num_spans_to_keep)
        marginalized_gold_scores = util.logsumexp(gold_scores, 2)
        log_norm = util.logsumexp(augmented_antecedent_scores, 2)
        negative_marginal_log_likelihood = log_norm - marginalized_gold_scores
        negative_marginal_log_likelihood = (negative_marginal_log_likelihood * top_span_mask.squeeze(-1)).sum()
        return negative_marginal_log_likelihood

    def _compute_antecedent_scores(self,
                                   pairwise_embeddings: torch.FloatTensor,
                                   top_span_mention_scores: torch.FloatTensor,
                                   antecedent_mention_scores: torch.FloatTensor,
                                   antecedent_log_mask: torch.FloatTensor):
        """
        Parameters
        ----------
        pairwise_embeddings: torch.FloatTensor
            Embedding representations of pairs of spans.
        top_span_mention_scores: torch.FloatTensor
            Mention scores for every span.
        antecedent_mention_scores: torch.FloatTensor
            Mention scores for every antecedent.
        antecedent_log_mask: torch.FloatTensor
            The log of the mask for valid antecedents.
        Returns
        -------
        augmented_antecedent_scores: torch.FloatTensor
            Scores for every pair of spans. For the dummy label, the score is always zero. For the true antecedent
            spans, the score consists of the pairwise antecedent score and the unary mention scores for the span
            and its antecedent. The factoring allows the model to blame many of the absent links on bad spans,
            enabling the pruning strategy used in the forward pass.
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(self._antecedent_feedforward(pairwise_embeddings)).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = Variable(antecedent_scores.data.new().resize_(shape).fill_(0),
                                requires_grad=False)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        augmented_antecedent_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return augmented_antecedent_scores

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                span_starts: torch.IntTensor,
                span_ends: torch.IntTensor,
                span_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor]
            From a ``TextField``
        span_starts : torch.IntTensor
            From a ``ListField[IndexField]``
        span_ends : torch.IntTensor
            From a ``ListField[IndexField]``
        span_labels : torch.IntTensor, optional (default = None)
            From a ``SequenceLabelField``
        Returns
        -------
        An output dictionary consisting of:
        top_spans : torch.IntTensor
            A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : torch.IntTensor
            A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents: torch.IntTensor
            A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, text_len, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        # Shape: (batch_size, text_len)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans, 1)
        span_mask = (span_starts >= 0).float()
        span_starts = F.relu(span_starts.float()).long()
        span_ends = F.relu(span_ends.float()).long()

        # Shape: (batch_size, num_spans, embedding_size)
        span_embeddings = self._compute_span_representations(text_embeddings,
                                                             text_mask,
                                                             span_starts,
                                                             span_ends)

        # Compute mention scores.
        # Shape: (batch_size, num_spans, 1)
        mention_scores = self._mention_scorer(self._mention_feedforward(span_embeddings))
        mention_scores += span_mask.log()

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * text_embeddings.size(1)))

        # Shape: (batch_size, num_spans_to_keep)
        top_span_indices = self._prune_and_sort_spans(mention_scores, num_spans_to_keep)

        # Shape: (batch_size * num_spans_to_keep)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, span_starts.size(1))

        # Select the span embeddings corresponding to the
        # top spans based on the mention scorer.
        # Shape: (batch_size, num_spans_to_keep, embedding_size)
        top_span_embeddings = util.batched_index_select(span_embeddings, top_span_indices, flat_top_span_indices)

        # Shape: (batch_size, num_spans_to_keep, 1)
        top_span_mask = util.batched_index_select(span_mask, top_span_indices, flat_top_span_indices)
        top_span_mention_scores = util.batched_index_select(mention_scores,
                                                            top_span_indices,
                                                            flat_top_span_indices)
        top_span_starts = util.batched_index_select(span_starts, top_span_indices, flat_top_span_indices)
        top_span_ends = util.batched_index_select(span_ends, top_span_indices, flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        antecedent_indices, antecedent_offsets, antecedent_log_mask = self._generate_antecedents(num_spans_to_keep,
                                                                                                 max_antecedents,
                                                                                                 text_mask.is_cuda)
        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_embeddings = util.flattened_index_select(top_span_embeddings, antecedent_indices)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
                                                                antecedent_indices).squeeze(-1)
        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        pairwise_embeddings = self._compute_pairwise_inputs(top_span_embeddings,
                                                            antecedent_embeddings,
                                                            antecedent_offsets)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        augmented_antecedent_scores = self._compute_antecedent_scores(pairwise_embeddings,
                                                                      top_span_mention_scores,
                                                                      antecedent_mention_scores,
                                                                      antecedent_log_mask)
        # Compute final predictions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = torch.cat([top_span_starts, top_span_ends], -1)

        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = augmented_antecedent_scores.max(2)
        predicted_antecedents -= 1

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": antecedent_indices,
                       "predicted_antecedents": predicted_antecedents}

        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            top_span_labels = util.batched_index_select(span_labels.unsqueeze(-1),
                                                        top_span_indices,
                                                        flat_top_span_indices)

            antecedent_labels = util.flattened_index_select(top_span_labels, antecedent_indices).squeeze(-1)
            antecedent_labels += antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            augmented_labels = self._compute_antecedent_gold_labels(top_span_labels, antecedent_labels)

            # Compute loss using the negative marginal log-likelihood.
            loss = self._compute_negative_marginal_log_likelihood(augmented_antecedent_scores,
                                                                  augmented_labels,
                                                                  top_span_mask)

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, antecedent_indices, predicted_antecedents, metadata)

            output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {"coref_precision": coref_precision,
                "coref_recall": coref_recall,
                "coref_f1": coref_f1,
                "mention_recall": mention_recall}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "CoreferenceResolver":
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        context_layer = Seq2SeqEncoder.from_params(params.pop("context_layer"))
        mention_feedforward = FeedForward.from_params(params.pop("mention_feedforward"))
        antecedent_feedforward = FeedForward.from_params(params.pop("antecedent_feedforward"))

        feature_size = params.pop("feature_size")
        max_span_width = params.pop("max_span_width")
        spans_per_word = params.pop("spans_per_word")
        max_antecedents = params.pop("max_antecedents")
        lexical_dropout = params.pop("lexical_dropout", 0.2)

        init_params = params.pop("initializer", None)
        reg_params = params.pop("regularizer", None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   context_layer=context_layer,
                   mention_feedforward=mention_feedforward,
                   antecedent_feedforward=antecedent_feedforward,
                   feature_size=feature_size,
                   max_span_width=max_span_width,
                   spans_per_word=spans_per_word,
                   max_antecedents=max_antecedents,
                   lexical_dropout=lexical_dropout,
                   initializer=initializer,
                   regularizer=regularizer)

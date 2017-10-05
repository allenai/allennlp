import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

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
        self._distance_embedding = Embedding(10, feature_size) # 10 possible distance buckets.
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

    def _compute_head_attention(self, head_scores, text_emb, span_ends, span_size):
        # Shape: (1, 1, max_span_width)
        head_offsets = util.get_range_vector(self._max_span_width, text_emb.is_cuda).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_span_width)
        head_mask = (head_offsets <= span_size).float()
        raw_head_idx = span_ends - head_offsets
        head_mask = head_mask * (raw_head_idx >= 0).float()
        head_idx = F.relu(raw_head_idx.float()).long()

        # Shape: (batch_size * num_spans * max_span_width)
        flat_head_idx = util.flatten_batched_indices(head_idx, text_emb.size(1))

        # Shape: (batch_size, num_spans, max_span_width, embedding_size)
        span_text_emb = util.batched_index_select(text_emb, head_idx, flat_head_idx)

        # Shape: (batch_size, num_spans, max_span_width)
        span_head_scores = util.batched_index_select(head_scores, head_idx, flat_head_idx).squeeze(-1)
        span_head_scores += head_mask.float().log()

        # Shape: (batch_size * num_spans, max_span_width)
        flat_span_head_scores = span_head_scores.view(-1, self._max_span_width)
        flat_span_head_weights = F.softmax(flat_span_head_scores)

        # Shape: (batch_size * num_spans, 1, max_span_width)
        flat_span_head_weights = flat_span_head_weights.unsqueeze(1)

        # Shape: (batch_size * num_spans, max_span_width, embedding_size)
        flat_span_text_emb = span_text_emb.view(-1, self._max_span_width, span_text_emb.size(-1))

        # Shape: (batch_size * num_spans, 1, embedding_size)
        flat_attended_text_emb = flat_span_head_weights.bmm(flat_span_text_emb)

        # Shape: (batch_size, num_spans, emb)
        attended_text_emb = flat_attended_text_emb.view(text_emb.size(0), span_ends.size(1), -1)
        return attended_text_emb

    def _compute_span_representations(self, text_emb, text_mask, span_starts, span_ends):
        # Shape: (batch_size, text_len, embedding_size)
        contextualized_emb = self._context_layer(text_emb, text_mask)

        # Shape: (batch_size, num_spans, embedding_size)
        start_emb = util.batched_index_select(contextualized_emb, span_starts.squeeze(-1))
        end_emb = util.batched_index_select(contextualized_emb, span_ends.squeeze(-1))

        # Shape: (batch_size, num_spans, 1)
        span_size = span_ends - span_starts # Strictly speaking span size minus one.

        # Shape: (batch_size, num_spans, embedding_size)
        span_size_emb = self._span_width_embedding(span_size.squeeze(-1))

        # Shape: (batch_size, text_len, 1)
        head_scores = self._head_scorer(contextualized_emb)

        # Shape: (batch_size, num_spans, embedding_size)
        attended_text_emb = self._compute_head_attention(head_scores, text_emb, span_ends, span_size)
        span_emb = torch.cat([start_emb, end_emb, span_size_emb, attended_text_emb], -1)
        return span_emb

    @staticmethod
    def _prune_and_sort_spans(mention_scores, k):
        # Shape: (batch_size, k, 1)
        _, top_span_idx = mention_scores.topk(k, 1)
        top_span_idx, _ = torch.sort(top_span_idx, 1)

        # Shape: (batch_size, k)
        top_span_idx = top_span_idx.squeeze(-1)
        return top_span_idx

    @staticmethod
    def _generate_antecedents(k, max_ant, is_cuda):
        # Shape: (k, 1)
        target_idx = util.get_range_vector(k, is_cuda).unsqueeze(1)

        # Shape: (1, max_ant)
        ant_offsets = (util.get_range_vector(max_ant, is_cuda) + 1).unsqueeze(0)

        # Shape: (k, max_ant)
        raw_ant_idx = target_idx - ant_offsets

        # Shape: (1, k, max_ant)
        ant_log_mask = (raw_ant_idx >= 0).float().unsqueeze(0).log()

        # Shape: (k, max_ant)
        ant_idx = F.relu(raw_ant_idx.float()).long()
        return ant_idx, ant_offsets, ant_log_mask

    def _compute_pairwise_inputs(self, top_span_emb, ant_emb, ant_offsets):
        # Shape: (batch_size, k max_ant, embedding_size)
        target_emb = top_span_emb.unsqueeze(2).expand_as(ant_emb)
        similarity_emb = ant_emb * target_emb

        # Shape: (1, max_ant, embedding_size)
        ant_distance_emb = self._distance_embedding(util.bucket_distance(ant_offsets))

        # Shape: (1, 1, max_ant, embedding_size)
        ant_distance_emb = ant_distance_emb.unsqueeze(0)

        # Shape: (batch_size, k, max_ant, embedding_size)
        ant_distance_emb = ant_distance_emb.expand(ant_emb.size(0),
                                                   ant_emb.size(1),
                                                   ant_emb.size(2),
                                                   ant_distance_emb.size(-1))

        # Shape: (batch_size, k, max_ant, embedding_size)
        pairwise_emb = torch.cat([target_emb,
                                  ant_emb,
                                  similarity_emb,
                                  ant_distance_emb], -1)
        return pairwise_emb

    @staticmethod
    def _compute_antecedent_labels(top_span_labels, ant_labels):
        # Shape: (batch_size, k, max_ant)
        target_labels = top_span_labels.expand_as(ant_labels)
        same_cluster_indicator = (target_labels == ant_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, k, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, k, max_ant + 1)
        augmented_labels = torch.cat([dummy_labels, pairwise_labels], -1) # [b, k, max_ant + 1]
        return augmented_labels

    @staticmethod
    def _compute_negative_marginal_loglikelihood(augmented_ant_scores, augmented_labels, top_span_mask):
        # Shape: (batch_size, k, max_ant + 1)
        gold_scores = augmented_ant_scores + augmented_labels.log()

        # Shape: (batch_size, k)
        marginalized_gold_scores = util.logsumexp(gold_scores, 2)
        log_norm = util.logsumexp(augmented_ant_scores, 2)
        nmll = log_norm - marginalized_gold_scores
        nmll = nmll * top_span_mask.squeeze(-1)
        return nmll.sum()

    def _compute_antecedent_scores(self, pairwise_emb, top_span_mention_scores, ant_mention_scores, ant_log_mask):
        # Shape: (batch_size, k, max_ant)
        ant_scores = self._antecedent_scorer(self._antecedent_feedforward(pairwise_emb)).squeeze(-1)
        ant_scores += top_span_mention_scores + ant_mention_scores # [b, k, max_ant]
        ant_scores += ant_log_mask # [b, k, max_ant]

        # Shape: (batch_size, k, 1)
        dummy_scores = util.get_zeros((ant_scores.size(0), ant_scores.size(1), 1), ant_scores.is_cuda) # [b, k, 1]

        # Shape: (batch_size, k, max_ant + 1)
        augmented_ant_scores = torch.cat([dummy_scores, ant_scores], -1)
        return augmented_ant_scores

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                span_starts: torch.IntTensor,
                span_ends: torch.IntTensor,
                span_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # Shape: (batch_size, text_len, embedding_size)
        text_emb = self._lexical_dropout(self._text_field_embedder(text))

        # Shape: (batch_size, text_len)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans, 1)
        span_mask = (span_starts >= 0).float()
        span_starts = F.relu(span_starts.float()).long()
        span_ends = F.relu(span_ends.float()).long()

        # Shape: (batch_size, num_spans, embedding_size)
        span_emb = self._compute_span_representations(text_emb,
                                                      text_mask,
                                                      span_starts,
                                                      span_ends)

        # Compute mention scores.
        # Shape: (batch_size, num_spans, 1)
        mention_scores = self._mention_scorer(self._mention_feedforward(span_emb))
        mention_scores += span_mask.log()

        # Prune based on mention scores.
        k = int(math.floor(self._spans_per_word * text_emb.size(1)))

        # Shape: (batch_size, k)
        top_span_idx = self._prune_and_sort_spans(mention_scores, k)

        # Shape: (batch_size * k)
        flat_top_span_idx = util.flatten_batched_indices(top_span_idx, span_starts.size(1))

        # Select tensors relating to the top spans.
        # Shape: (batch_size, k, embedding_size)
        top_span_emb = util.batched_index_select(span_emb, top_span_idx, flat_top_span_idx)

        # Shape: (batch_size, k, 1)
        top_span_mask = util.batched_index_select(span_mask, top_span_idx, flat_top_span_idx)
        top_span_mention_scores = util.batched_index_select(mention_scores, top_span_idx, flat_top_span_idx)
        top_span_starts = util.batched_index_select(span_starts, top_span_idx, flat_top_span_idx)
        top_span_ends = util.batched_index_select(span_ends, top_span_idx, flat_top_span_idx)
        if span_labels is not None:
            top_span_labels = util.batched_index_select(span_labels.unsqueeze(-1),
                                                        top_span_idx,
                                                        flat_top_span_idx)

        # Compute indices for antecedent spans to consider.
        max_ant = min(self._max_antecedents, k)

        # Shapes: (k, max_ant), (1, max_ant), (1, k, max_ant)
        ant_idx, ant_offsets, ant_log_mask = self._generate_antecedents(k, max_ant, text_emb.is_cuda)

        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, k, embedding_size)
        ant_emb = util.flattened_index_select(top_span_emb, ant_idx) # [b, k, max_ant, emb]

        # Shape: (batch_size, k, max_ant)
        ant_mention_scores = util.flattened_index_select(top_span_mention_scores, ant_idx).squeeze(-1)
        if span_labels is not None:
            ant_labels = util.flattened_index_select(top_span_labels, ant_idx).squeeze(-1)
            ant_labels += ant_log_mask.long()

        # Compute antecedent scores.
        # Shape: (batch_size, k, max_ant, embedding_size)
        pairwise_emb = self._compute_pairwise_inputs(top_span_emb, ant_emb, ant_offsets)

        # Shape: (batch_size, k, max_ant + 1)
        augmented_ant_scores = self._compute_antecedent_scores(pairwise_emb,
                                                               top_span_mention_scores,
                                                               ant_mention_scores,
                                                               ant_log_mask)

        # Compute final predictions.
        # Shape: (batch_size, k, 2)
        top_spans = torch.cat([top_span_starts, top_span_ends], -1)

        # Shape: (batch_size, k)
        _, predicted_ants = augmented_ant_scores.max(2)
        predicted_ants -= 1

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": ant_idx,
                       "predicted_antecedents": predicted_ants}

        if span_labels is not None:
            # Compute labels.
            # Shape: (batch_size, k, max_ant + 1)
            augmented_labels = self._compute_antecedent_labels(top_span_labels, ant_labels)

            # Compute loss using the negative marginal log-likelihood.
            loss = self._compute_negative_marginal_loglikelihood(augmented_ant_scores,
                                                                 augmented_labels,
                                                                 top_span_mask)

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, ant_idx, predicted_ants, metadata)

            output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)
        return {"coref_precision" : coref_precision,
                "coref_recall" : coref_recall,
                "coref_f1" : coref_f1,
                "mention_recall" : mention_recall}

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

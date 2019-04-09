import logging
from allennlp.common.elastic_logger import ElasticLogger
from typing import Any, Dict, List
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
import os
import random
import traceback
import json

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn import InitializerApplicator, util
from allennlp.tools import squad_eval
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("BERT_QA")
class BERT_QA(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 max_span_length: int = 30,
                 predictions_file = None,
                 use_multi_label_loss: bool = False,
                 stats_report_freq:float = None,
                 debug_experiment_name:str = None) -> None:
        super().__init__(vocab)
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._stats_report_freq = stats_report_freq
        self._debug_experiment_name = debug_experiment_name
        self._use_multi_label_loss = use_multi_label_loss
        self._predictions_file = predictions_file

        # TODO get rid of this patch...
        if predictions_file is not None and os.path.isfile(predictions_file):
            os.remove(predictions_file)

        # see usage below for explanation
        self._all_qa_count = 0
        self._qas_used_fraction = 1.0
        self.qa_outputs = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 2)

        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._official_f1 = Average()
        self._official_EM = Average()
        self._variational_dropout = InputVariationalDropout(dropout)

    def multi_label_cross_entropy_loss(self, span_logits, answers, passage_length):
        instances_with_answer = np.argwhere(answers.squeeze().cpu() >= 0)[0].unique()
        target = torch.cuda.FloatTensor(len(instances_with_answer), passage_length, device=span_logits.device) \
            if torch.cuda.is_available() else torch.FloatTensor(len(instances_with_answer), passage_length)
        target.zero_()

        answers = answers[instances_with_answer].squeeze().cpu() if len(instances_with_answer)>1 \
            else answers[instances_with_answer].cpu()

        for ind, q_target in enumerate(answers):
            target[ind, q_target[(q_target >= 0) & (q_target < passage_length)]] = 1.0

        return -(torch.log((F.softmax(span_logits[instances_with_answer], dim=-1) * \
                            target.float()).sum(dim=1))).mean()

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, num_of_passage_tokens = passage['bert'].size()

        embedded_passage = self._text_field_embedder(passage)
        passage_length = embedded_passage.size(1)
        logits = self.qa_outputs(embedded_passage)
        start_logits, end_logits = logits.split(1, dim=-1)
        span_start_logits = start_logits.squeeze(-1)
        span_end_logits = end_logits.squeeze(-1)

        passage_mask = util.get_text_field_mask(passage).float()
        repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, 1, 1)
        repeated_passage_mask = repeated_passage_mask.view(batch_size, passage_length)

        span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)

        best_span = self._get_example_predications(span_start_logits, span_end_logits,self._max_span_length)

        output_dict: Dict[str, Any] = {}

        intances_question_id = [insta_meta['question_id'] for insta_meta in metadata]
        question_instances_split_inds = np.cumsum(np.unique(intances_question_id, return_counts=True)[1])[:-1]
        per_question_inds = np.split(range(batch_size), question_instances_split_inds)
        metadata = np.split(metadata, question_instances_split_inds)

        # Compute the loss.
        if span_start is not None and len(np.argwhere(span_start.squeeze().cpu() >= 0)) > 0:
            # Per instance loss
            if self._use_multi_label_loss:
                try:
                    loss = self.multi_label_cross_entropy_loss(span_start_logits, span_start, passage_length)
                    loss += self.multi_label_cross_entropy_loss(span_end_logits, span_end, passage_length)
                    output_dict["loss"] = loss
                except:
                    ElasticLogger().write_log('INFO', 'Loss Error', context_dict={'span_start_logits':span_start_logits.cpu().size(),
                        'span_end_logits_size':span_end_logits.cpu().size(),'span_start':span_start.squeeze().cpu().numpy().tolist(),
                            'span_end':span_end.squeeze().cpu().numpy().tolist(),'error_message': traceback.format_exc(),
                                                                'batch_size':batch_size, 'passage_length':passage_length},print_log=True)
                    a = torch.autograd.Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
                    loss = torch.sum(a ** 2)
                    output_dict["loss"] = loss
            else:

                inds_with_gold_answer = np.argwhere(span_start.view(-1).cpu().numpy() >= 0)
                inds_with_gold_answer = inds_with_gold_answer.squeeze() if len(inds_with_gold_answer) > 1 else inds_with_gold_answer
                if len(inds_with_gold_answer)>0:
                    loss = nll_loss(util.masked_log_softmax(span_start_logits[inds_with_gold_answer], \
                                                        repeated_passage_mask[inds_with_gold_answer]),\
                                    span_start.view(-1)[inds_with_gold_answer], ignore_index=-1)
                    loss += nll_loss(util.masked_log_softmax(span_end_logits[inds_with_gold_answer], \
                                                        repeated_passage_mask[inds_with_gold_answer]),\
                                    span_end.view(-1)[inds_with_gold_answer], ignore_index=-1)
                    output_dict["loss"] = loss


        span_start_logits_numpy = span_start_logits.data.cpu().numpy()
        span_end_logits_numpy = span_end_logits.data.cpu().numpy()


        # Compute F1 and preparing the output dictionary.
        output_dict['best_span_str'] = []
        output_dict['qid'] = []

        # best_span is a vector of more than one span
        best_span_cpu = best_span.detach().cpu().numpy()

        # Iterating over every question (which may contain multiple instances, one per chunk)
        for question_inds, question_instances_metadata in zip(per_question_inds, metadata):
            if len(question_inds) == 0:
                continue

            # We need to perform softmax here !!
            best_span_ind = np.argmax(span_start_logits_numpy[question_inds, best_span_cpu[question_inds][:, 0]] +
                      span_end_logits_numpy[question_inds, best_span_cpu[question_inds][:, 1]])
            best_span_logit = np.max(span_start_logits_numpy[question_inds, best_span_cpu[question_inds][:, 0]] +
                                      span_end_logits_numpy[question_inds, best_span_cpu[question_inds][:, 1]])

            passage_str = question_instances_metadata[best_span_ind]['original_passage']
            offsets = question_instances_metadata[best_span_ind]['token_offsets']

            predicted_span = best_span_cpu[question_inds[best_span_ind]]
            start_offset = offsets[predicted_span[0]][0]
            end_offset = offsets[predicted_span[1]][1]
            best_span_string = passage_str[start_offset:end_offset]

            f1_score = 0.0
            EM_score = 0.0
            gold_answer_texts = question_instances_metadata[best_span_ind]['answer_texts_list']
            if gold_answer_texts:
                f1_score = squad_eval.metric_max_over_ground_truths(squad_eval.f1_score,best_span_string,gold_answer_texts)
                EM_score = squad_eval.metric_max_over_ground_truths(squad_eval.exact_match_score, best_span_string,gold_answer_texts)
            self._official_f1(100 * f1_score)
            self._official_EM(100 * EM_score)

            if self._predictions_file is not None:
                with open(self._predictions_file,'a') as f:
                    f.write(json.dumps({'question_id':question_instances_metadata[best_span_ind]['question_id'], \
                                'best_span_logit':float(best_span_logit), \
                                'f1':100 * f1_score,
                                'EM':100 * EM_score,
                                'best_span_string':best_span_string,\
                                'gold_answer_texts':gold_answer_texts, \
                                'qas_used_fraction':1.0}) + '\n')
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'EM': self._official_EM.get_metric(reset),
                'f1': self._official_f1.get_metric(reset),
                'qas_used_fraction': 1.0}


    @staticmethod
    def _get_example_predications(span_start_logits: torch.Tensor,
                                      span_end_logits: torch.Tensor,
                                      max_span_length: int) -> torch.Tensor:
        # Returns the index of highest-scoring span that is not longer than 30 tokens, as well as
        # yesno prediction bit and followup prediction bit from the predicted span end token.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size

        best_word_span = span_start_logits.new_zeros((batch_size, 4), dtype=torch.long)

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()
        for b_i in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b_i, span_start_argmax[b_i]]
                if val1 < span_start_logits[b_i, j]:
                    span_start_argmax[b_i] = j
                    val1 = span_start_logits[b_i, j]
                val2 = span_end_logits[b_i, j]
                if val1 + val2 > max_span_log_prob[b_i]:
                    if j - span_start_argmax[b_i] > max_span_length:
                        continue
                    best_word_span[b_i, 0] = span_start_argmax[b_i]
                    best_word_span[b_i, 1] = j
                    max_span_log_prob[b_i] = val1 + val2
        for b_i in range(batch_size):
            j = best_word_span[b_i, 1]

        return best_word_span

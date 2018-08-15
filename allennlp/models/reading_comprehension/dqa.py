import numpy as np
import logging
from typing import Any, Dict, List
import json
import torch
from overrides import overrides
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import nll_loss

from allennlp.common import Params, squad_eval
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.tri_linear_attention import TriLinearAttention
from allennlp.modules.variational_dropout import VariationalDropout
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dqa")
class DQA(Model):
  """
  This class implements modified version of BiDAF (with self attention and residual layer, from Clark and Gardner ACL 17 paper) model as used in 
  Question Answering in Context (EMNLP 2018) Paper.

  Parameters
  ----------
  vocab : ``Vocabulary``
  text_field_embedder : ``TextFieldEmbedder``
      Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
  phrase_layer : ``Seq2SeqEncoder``
      The encoder (with its own internal stacking) that we will use in between embedding tokens
      and doing the bidirectional attention.
  span_start_encoder : ``Seq2SeqEncoder``
      The encoder that we will use to incorporate span start predictions into the passage state
      before predicting span end.
  span_end_encoder : ``Seq2SeqEncoder``
      The encoder that we will use to incorporate span end predictions into the passage state.
  dropout : ``float``, optional (default=0.2)
      If greater than 0, we will apply dropout with this probability after all encoders (pytorch
      LSTMs do not apply dropout to their last layer).
  prev_a : ``int``, optional (default=0)
      If greater than 0, the model will consider previous question answering context.
  mask_lstms : ``bool``, optional (default=True)
      If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
      with only a slight performance decrease, if any.  We haven't experimented much with this
      yet, but have confirmed that we still get very similar performance with much faster
      training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
      required when using masking with pytorch LSTMs.
  """
  def __init__(self, vocab: Vocabulary,
               text_field_embedder: TextFieldEmbedder,
               phrase_layer: Seq2SeqEncoder,
               residual_encoder: Seq2SeqEncoder,
               span_start_encoder: Seq2SeqEncoder,
               span_end_encoder: Seq2SeqEncoder,
               initializer: InitializerApplicator,
               dropout: float = 0.2,
               prev_a: int = 0,
               mask_lstms: bool = True) -> None:
    super(DQA, self).__init__(vocab)
    self._prev_a = prev_a
    self._text_field_embedder = text_field_embedder
    self._phrase_layer = phrase_layer
    self._matrix_attention = TriLinearAttention(200)
    self._merge_atten = TimeDistributed(torch.nn.Linear(200 * 4, 200))

    self._residual_encoder = residual_encoder
    self._prev_ans_marker = torch.nn.Embedding((prev_a * 4 * 3) + 1, 10)
    self._question_num_marker = torch.nn.Embedding(12, 10 * prev_a)
    
    self._self_atten = TriLinearAttention(200)

    self._followup_lin = torch.nn.Linear(200, 3)
    self._merge_self_atten = TimeDistributed(torch.nn.Linear(200 * 3, 200))

    self._span_start_encoder = span_start_encoder
    self._span_end_encoder = span_end_encoder

    self._span_start_predictor = TimeDistributed(torch.nn.Linear(200, 1))
    self._span_end_predictor = TimeDistributed(torch.nn.Linear(200, 1))
    self._span_yesno_predictor = TimeDistributed(torch.nn.Linear(200, 3))
    self._span_followup_predictor = TimeDistributed(self._followup_lin)

    initializer(self)

    self._span_start_accuracy = CategoricalAccuracy()
    self._span_end_accuracy = CategoricalAccuracy()
    self._span_yesno_accuracy = CategoricalAccuracy()
    self._span_followup_accuracy = CategoricalAccuracy()

    self._span_gt_yesno_accuracy = CategoricalAccuracy()
    self._span_gt_followup_accuracy = CategoricalAccuracy()

    self._span_accuracy = BooleanAccuracy()
    self._official_f1 = Average()
    if dropout > 0:
      self._dropout = VariationalDropout(p=dropout)
    else:
      raise ValueError()
    self._mask_lstms = mask_lstms
  
  def forward(self,  # type: ignore
              question: Dict[str, torch.LongTensor],
              passage: Dict[str, torch.LongTensor],
              span_start: torch.IntTensor = None,
              span_end: torch.IntTensor = None,
              p1_answer_marker: torch.IntTensor = None,
              p2_answer_marker: torch.IntTensor = None,
              p3_answer_marker: torch.IntTensor = None,
              yesno_list: torch.IntTensor = None,
              followup_list: torch.IntTensor = None,
              metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    model_in_cuda = span_start.is_cuda
    qa_mask = (1 - torch.eq(followup_list, -1)).view(-1)  # not all dialog has the same number of QA pairs. 
    batch_size, max_qa_count, max_q_len, max_word_len = question['token_characters'].size()
    #_, _, token_max_q_len = question['tokens'].size()
    #assert token_max_q_len == max_q_len
    question['token_characters'] = question['token_characters'].view(batch_size * max_qa_count, max_q_len, -1)
    question['tokens'] = question['tokens'].view(batch_size * max_qa_count, max_q_len)
    embedded_question = self._dropout(self._text_field_embedder(question))
    embedded_passage = self._dropout(self._text_field_embedder(passage))
    passage_length = embedded_passage.size(1)

    question_mask = util.get_text_field_mask(question).float()
    passage_mask = util.get_text_field_mask(passage).float()
  
    question_lstm_mask = question_mask if self._mask_lstms else None
    passage_lstm_mask = passage_mask if self._mask_lstms else None
    
    if self._prev_a > 0:
      question_num_ind = torch.Tensor(list(range(0, max_qa_count))* batch_size).long().reshape(-1, 1).repeat(1, max_q_len)
      if model_in_cuda:
        question_num_ind.cuda()
      question_num_marker_emb = self._question_num_marker(question_num_ind)
      embedded_question = torch.cat([embedded_question, question_num_marker_emb], dim=-1)
      repeated_embedded_passage = embedded_passage. \
      		unsqueeze(1).repeat(1, max_qa_count, 1, 1).view(batch_size * max_qa_count, passage_lstm_mask.size()[1], -1)  # batch_size * max_qa_count, passage_length, word_embed_dim
      p1_answer_marker = p1_answer_marker.view(batch_size * max_qa_count, passage_length)
      p1_answer_marker_emb = self._prev_ans_marker(p1_answer_marker)
      repeated_embedded_passage = torch.cat([repeated_embedded_passage, p1_answer_marker_emb], dim=-1)
      if self._prev_a > 1:
        p2_answer_marker = p2_answer_marker.view(batch_size * max_qa_count, passage_length)
        p2_answer_marker_emb = self._prev_ans_marker(p2_answer_marker)
        repeated_embedded_passage = torch.cat([repeated_embedded_passage, p2_answer_marker_emb], dim=-1)
        if self._prev_a > 2:
          p3_answer_marker = p3_answer_marker.view(batch_size * max_qa_count, passage_length)
          p3_answer_marker_emb = self._prev_ans_marker(p3_answer_marker)
          repeated_embedded_passage = torch.cat([repeated_embedded_passage, p3_answer_marker_emb], dim=-1)
      repeated_passage_lstm_mask = passage_lstm_mask.unsqueeze(1).repeat(1, max_qa_count, 1).view(batch_size * max_qa_count, passage_length)
      repeated_encoded_passage = self._dropout(self._phrase_layer(repeated_embedded_passage, repeated_passage_lstm_mask))
    else:
      encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
      repeated_encoded_passage = encoded_passage.unsqueeze(1).repeat(1, max_qa_count, 1, 1).view(batch_size * max_qa_count, passage_lstm_mask.size()[1], -1)

    encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
    encoding_dim = encoded_question.size(-1)

    # Shape: (batch_size * max_qa_count, passage_length, question_length)
    passage_question_similarity = self._matrix_attention(repeated_encoded_passage, encoded_question)
    # Shape: (batch_size * max_qa_count, passage_length, question_length)
    passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)

    # Shape: (batch_size * max_qa_count, passage_length, encoding_dim)
    passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

    # We replace masked values with something really negative here, so they don't affect the
    # max below.
    masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                   question_mask.unsqueeze(1),
                                                   -1e7)

    passage_mask = passage_mask.unsqueeze(1).repeat(1, max_qa_count, 1).view(batch_size * max_qa_count, -1)
    question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
    question_passage_attention = util.last_dim_softmax(question_passage_similarity,
                                                       passage_mask)
    # Shape: (batch_size * max_qa_count, encoding_dim)
    question_passage_vector = util.weighted_sum(repeated_encoded_passage, question_passage_attention)
    tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size * max_qa_count,
                                                                                passage_length,
                                                                                encoding_dim)

    # Shape: (batch_size * max_qa_count, passage_length, encoding_dim * 4)
    final_merged_passage = torch.cat([repeated_encoded_passage,
                                      passage_question_vectors,
                                      repeated_encoded_passage * passage_question_vectors,
                                      repeated_encoded_passage * tiled_question_passage_vector],
                                     dim=-1)

    final_merged_passage = F.relu(self._merge_atten(final_merged_passage))

    residual_layer = self._dropout(self._residual_encoder(self._dropout(final_merged_passage), passage_mask))
    self_atten_matrix = self._self_atten(residual_layer, residual_layer)

    mask = passage_mask.resize(batch_size * max_qa_count, passage_length, 1) * passage_mask.resize(
      batch_size * max_qa_count, 1, passage_length)

    # torch.eye does not have a gpu implementation, so we are forced to use the cpu one and .cuda()
    # Not sure if this matters for performance
    self_mask = Variable(torch.eye(passage_length, passage_length)).resize(1, passage_length, passage_length)
    if model_in_cuda:
      self_mask = self_mask.cuda()
    # self_mask = Variable(torch.eye(passage_length, passage_length)).resize(1, passage_length, passage_length)
    mask = mask * (1 - self_mask)

    self_atten_probs = util.last_dim_softmax(self_atten_matrix, mask)

    # Batch matrix multiplication:
    # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
    self_atten_vecs = torch.matmul(self_atten_probs, residual_layer)

    residual_layer = F.relu(self._merge_self_atten(torch.cat(
      [self_atten_vecs, residual_layer, residual_layer * self_atten_vecs], dim=-1)))

    final_merged_passage += residual_layer  # batch_size * maxqa_pair_len * max_passage_len * 200
    final_merged_passage = self._dropout(final_merged_passage)
    start_rep = self._span_start_encoder(final_merged_passage, passage_mask)
    span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)
    span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

    end_rep = self._span_end_encoder(torch.cat([final_merged_passage, start_rep], dim=-1), passage_mask)
    span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)
    span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

    span_yesno_logits = self._span_yesno_predictor(end_rep).squeeze(-1)
    span_followup_logits = self._span_followup_predictor(end_rep).squeeze(-1)

    span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)  # batch_size * maxqa_len_pair, max_document_len
    span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)

    best_span = self._get_best_span(span_start_logits, span_end_logits, span_yesno_logits, span_followup_logits)

    output_dict = {}
    # Compute the loss.
    if span_start is not None:
      loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.view(-1), ignore_index=-1)
      self._span_start_accuracy(span_start_logits, span_start.view(-1), mask=qa_mask)
      loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.view(-1), ignore_index=-1)
      self._span_end_accuracy(span_end_logits, span_end.view(-1), mask=qa_mask)
      self._span_accuracy(best_span[:, 0:2], torch.stack([span_start, span_end], -1).view(batch_size * max_qa_count, 2), mask=qa_mask.unsqueeze(1).expand(-1, 2).long())
      # add a select for the right span to compute loss
      v = []
      span_end = span_end.view(-1).squeeze().data.cpu().numpy()
      for i in range(0, batch_size * max_qa_count):
        v.append(max(span_end[i] * 3 + i * passage_length * 3, 0))
        v.append(max(span_end[i] * 3 + i * passage_length * 3 + 1, 0))
        v.append(max(span_end[i] * 3 + i * passage_length * 3 + 2, 0))
      gt_end = Variable(torch.LongTensor(v))
      if model_in_cuda:
        gt_end = gt_end.cuda()

      v = []
      for i in range(0, batch_size * max_qa_count):
        v.append(max(best_span[i][1] * 3 + i * passage_length * 3, 0))
        v.append(max(best_span[i][1] * 3 + i * passage_length * 3 + 1, 0))
        v.append(max(best_span[i][1] * 3 + i * passage_length * 3 + 2, 0))
      predicted_end = Variable(torch.LongTensor(v))
      if model_in_cuda:
        predicted_end = predicted_end.cuda()

      _yesno = span_yesno_logits.view(-1).index_select(0, gt_end).view(-1, 3)
      _followup = span_followup_logits.view(-1).index_select(0, gt_end).view(-1, 3)
      loss += nll_loss(F.log_softmax(_yesno, dim=-1), yesno_list.view(-1), ignore_index=-1)
      loss += nll_loss(F.log_softmax(_followup, dim=-1), followup_list.view(-1), ignore_index=-1)
      
      _yesno = span_yesno_logits.view(-1).index_select(0, predicted_end).view(-1, 3)
      _followup = span_followup_logits.view(-1).index_select(0, predicted_end).view(-1, 3)
      self._span_yesno_accuracy(_yesno, yesno_list.view(-1), mask=qa_mask)
      self._span_followup_accuracy(_followup, followup_list.view(-1), mask=qa_mask)
      output_dict['best_span_str'] = []
      output_dict['qid'] = []
      output_dict['aid'] = []

      # Eval
      best_span_cpu = best_span.detach().cpu().numpy()
      for i in range(batch_size):
        passage_str = metadata[i]['original_passage']
        offsets = metadata[i]['token_offsets']
        f1_score = 0
        output_bspan_list = []
        output_q_list = []
        output_aid_list = []
        for currcount, (iid, answer_texts) in enumerate(
                zip(metadata[i]["instance_id"], metadata[i]["answer_texts_list"])):
          (aid, _) = iid.split("_q#")
          predicted_span = tuple(best_span_cpu[i * max_qa_count + currcount])
          start_offset = offsets[predicted_span[0]][0]
          end_offset = offsets[predicted_span[1]][1]
          best_span_string = passage_str[start_offset:end_offset]
          output_bspan_list.append(best_span_string)
          output_q_list.append(iid)
          output_aid_list.append(aid)
          if answer_texts:
            if len(answer_texts) > 1:
              t_f1 = []
              for ai in range(len(answer_texts)):
                idxes = list(range(len(answer_texts)))
                idxes.pop(ai)
                refs = [answer_texts[z] for z in idxes]
                t_f1.append(squad_eval.metric_max_over_ground_truths(
                  squad_eval.f1_score,
                  best_span_string,
                  refs))
              f1_score = 1.0 * sum(t_f1)/len(t_f1)
            else:
              f1_score = squad_eval.metric_max_over_ground_truths(
              squad_eval.f1_score,
              best_span_string,
              answer_texts)
          self._official_f1(100 * f1_score)
        output_dict['qid'].append(output_q_list)
        output_dict['aid'].append(output_aid_list)
        output_dict['best_span_str'].append(output_bspan_list)
    output_dict["loss"] = loss
    output_dict['yesno'] = _yesno.view(batch_size, -1, 3).detach().cpu().numpy()
    return output_dict 

  @overrides
  def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    yes_no_cpu = output_dict.pop('yesno')
    argmax_indices = np.argmax(yes_no_cpu, axis=-1)
    yesno_tags = [[self.vocab.get_token_from_index(x, namespace="yesno_labels") for x in argm_list] for argm_list in argmax_indices]
    output_dict['yesno']  = yesno_tags
    return output_dict

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {
      'start_acc': self._span_start_accuracy.get_metric(reset),
      'end_acc': self._span_end_accuracy.get_metric(reset),
      'span_acc': self._span_accuracy.get_metric(reset),
      'yesno': self._span_yesno_accuracy.get_metric(reset),
      'followup': self._span_followup_accuracy.get_metric(reset),
      'f1': self._official_f1.get_metric(reset),
    }

  @staticmethod
  def _get_best_span(span_start_logits: Variable, span_end_logits: Variable, span_yesno_logits: Variable,
                     span_followup_logits: Variable) -> Variable:
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
      raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    max_span_log_prob = [-1e20] * batch_size
    span_start_argmax = [0] * batch_size
    best_word_span = Variable(span_start_logits.data.new()
                              .resize_(batch_size, 4).fill_(0)).long()

    span_start_logits = span_start_logits.data.cpu().numpy()
    span_end_logits = span_end_logits.data.cpu().numpy()
    span_yesno_logits = span_yesno_logits.data.cpu().numpy()
    span_followup_logits = span_followup_logits.data.cpu().numpy()

    for b in range(batch_size):  # pylint: disable=invalid-name
      for j in range(passage_length):
        val1 = span_start_logits[b, span_start_argmax[b]]
        if val1 < span_start_logits[b, j]:
          span_start_argmax[b] = j
          val1 = span_start_logits[b, j]
        val2 = span_end_logits[b, j]
        if val1 + val2 > max_span_log_prob[b]:
          if j - span_start_argmax[b] > 30:
            continue 
          best_word_span[b, 0] = span_start_argmax[b]
          best_word_span[b, 1] = j
          max_span_log_prob[b] = val1 + val2
    for b in range(batch_size):
      j = best_word_span[b, 1]
      yn = np.argmax(span_yesno_logits[b, j])
      fu = np.argmax(span_followup_logits[b, j])
      best_word_span[b, 2] = int(yn)
    return best_word_span

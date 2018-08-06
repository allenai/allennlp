import numpy as np
import logging
from typing import Any, Dict, List

import torch
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


@Model.register("bidaf-self-atten-bits")
class BidafPlusSelfAttentionBits(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 residual_encoder: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 outfile: str = None) -> None:
        super(BidafPlusSelfAttentionBits, self).__init__(vocab)
        outfile = './wdev_l.txt'
        if outfile is not None: 
          self.write_out = True
          self.outfile = outfile
          print("appending output to {}".format(outfile))
        else:
          self.write_out = False
          self.outfile = None
        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer
        self._matrix_attention = TriLinearAttention(200)

        self._merge_atten = TimeDistributed(torch.nn.Linear(200 * 4, 200))

        self._residual_encoder = residual_encoder
        self._self_atten = TriLinearAttention(200)

        self._followup_lin = torch.nn.Linear(200, 4)
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
        self._official_em = Average()
        self._official_f1 = Average()
        if dropout > 0:
            # self._dropout = torch.nn.Dropout(p=dropout)
            self._dropout = VariationalDropout(p=dropout)
        else:
            raise ValueError()
            # self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                yesno: torch.IntTensor = None,
                followup: torch.IntTensor = None,
                prev_followup: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` index.  If
            this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` index.  If
            this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)
        #prev_followup_emb = self._followup_emb(prev_followup)


        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)
        #tiled_prev_followup = prev_followup_emb.expand(batch_size, passage_length, 200)
        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        final_merged_passage = F.relu(self._merge_atten(final_merged_passage))

        residual_layer = self._dropout(self._residual_encoder(self._dropout(final_merged_passage), passage_mask))
        self_atten_matrix = self._self_atten(residual_layer, residual_layer)

        mask = passage_mask.resize(batch_size, passage_length, 1) * passage_mask.resize(batch_size, 1, passage_length)

        # torch.eye does not have a gpu implementation, so we are forced to use the cpu one and .cuda()
        # Not sure if this matters for performance
        self_mask = Variable(torch.eye(passage_length, passage_length).cuda()).resize(1, passage_length, passage_length)
        mask = mask * (1 - self_mask)

        self_atten_probs = util.last_dim_softmax(self_atten_matrix, mask)

        # Batch matrix multiplication:
        # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
        self_atten_vecs = torch.matmul(self_atten_probs, residual_layer)

        residual_layer = F.relu(self._merge_self_atten(torch.cat(
            [self_atten_vecs, residual_layer, residual_layer * self_atten_vecs], dim=-1)))

        final_merged_passage += residual_layer

        final_merged_passage = self._dropout(final_merged_passage)

        start_rep = self._span_start_encoder(final_merged_passage, passage_lstm_mask)
        span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        end_rep = self._span_end_encoder(torch.cat([final_merged_passage, start_rep], dim=-1), passage_lstm_mask)
        span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        span_yesno_logits = self._span_yesno_predictor(end_rep).squeeze(-1)
        span_followup_logits = self._span_followup_predictor(end_rep).squeeze(-1)
        
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)

        best_span = self._get_best_span(span_start_logits, span_end_logits, span_yesno_logits, span_followup_logits)

        output_dict = {"span_start_logits": span_start_logits,
                       "span_start_probs": span_start_probs,
                       "span_end_logits": span_end_logits,
                       "span_end_probs": span_end_probs,
                       "span_yesno_logits": span_yesno_logits,
                       "span_followup_logits": span_followup_logits,
                       "best_span": best_span}
        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span[:,0:2], torch.stack([span_start, span_end], -1))
            #add a select for the right span
            v = []
            _span_end = span_end.squeeze()
            for i in range(0, batch_size):
              v.append(span_end[i]*3 + i*passage_length*3)
              v.append(span_end[i]*3 + i*passage_length*3 + 1)
              v.append(span_end[i]*3 + i*passage_length*3 + 2)
            gt_end = Variable(torch.LongTensor(v).cuda())

            v = []
            for i in range(0, batch_size):
              v.append(best_span[i][1]*3 + i*passage_length*3)
              v.append(best_span[i][1]*3 + i*passage_length*3 + 1)
              v.append(best_span[i][1]*3 + i*passage_length*3 + 2)


            predicted_end = Variable(torch.LongTensor(v).cuda())

            _yesno = span_yesno_logits.view(-1).index_select(0,gt_end).view(-1,3)
            _followup = span_followup_logits.view(-1).index_select(0,gt_end).view(-1,3)
            loss += nll_loss(F.log_softmax(_yesno,dim=-1), yesno.squeeze(-1))
            loss += nll_loss(F.log_softmax(_followup,dim=-1), followup.squeeze(-1))
 
            self._span_gt_yesno_accuracy(_yesno, yesno.squeeze(-1))
            self._span_gt_followup_accuracy(_followup, followup.squeeze(-1))          
            
            _yesno = span_yesno_logits.view(-1).index_select(0,predicted_end).view(-1,3)
            _followup = span_followup_logits.view(-1).index_select(0,predicted_end).view(-1,3)

            self._span_yesno_accuracy(_yesno, yesno.squeeze(-1))
            self._span_followup_accuracy(_followup, followup.squeeze(-1))

            #add a select for the right span
            output_dict["loss"] = loss

        def clean(s):
          for i in range(0,10):
            prepend = "ANSWER{}_START".format(i)
            postpend = "ANSWER{}_END".format(i)
            s = s.replace(prepend, "").replace(postpend, "")
          return s

        if metadata is not None:
            output_dict['best_span_str'] = []
            if self.outfile is not None:
              f = open(self.outfile, "a")
            for i in range(batch_size):
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].data.cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                exact_match = f1_score = 0
                
                if self.outfile is not None:
                  (aid, qid) = metadata[i]["instance_id"].split("_q#")
                  f.write("{}\t{}\t{}\t{}\t{}\n".format(aid, qid, metadata[i]["question"], clean(best_span_string), answer_texts)) 

                if answer_texts:
                    exact_match = squad_eval.metric_max_over_ground_truths(
                            squad_eval.exact_match_score,
                            best_span_string,
                            answer_texts)
                    f1_score = squad_eval.metric_max_over_ground_truths(
                            squad_eval.f1_score,
                            best_span_string,
                            answer_texts)
                self._official_em(100 * exact_match)
                self._official_f1(100 * f1_score)
        if self.outfile is not None:
          f.close()
        
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'yesno' : self._span_yesno_accuracy.get_metric(reset),
                'followup': self._span_followup_accuracy.get_metric(reset),
                'gt_yesno' : self._span_gt_yesno_accuracy.get_metric(reset),
                'gt_followup': self._span_gt_followup_accuracy.get_metric(reset),
                'em': self._official_em.get_metric(reset),
                'f1': self._official_f1.get_metric(reset),
                }

    @staticmethod
    def _get_best_span(span_start_logits: Variable, span_end_logits: Variable, span_yesno_logits:Variable, span_followup_logits: Variable) -> Variable:
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
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        for b in range(batch_size):
            j = best_word_span[b, 1]
            yn = np.argmax(span_yesno_logits[b,j])
            fu = np.argmax(span_followup_logits[b,j])
            best_word_span[b,2] = int(yn)

        return best_word_span

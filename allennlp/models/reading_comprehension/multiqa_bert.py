import logging
from typing import Any, Dict, List
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
from torch.nn.functional import cross_entropy
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.tools import squad_eval
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("multiqa_bert")
class MultiQA_BERT(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator,
                 max_span_length: int = 30,
                 use_multi_label_loss: bool = False,
                 stats_report_freq:float = None,
                 debug_experiment_name:str = None) -> None:
        super().__init__(vocab)
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._stats_report_freq = stats_report_freq
        self._debug_experiment_name = debug_experiment_name
        self._use_multi_label_loss = use_multi_label_loss


        # see usage below for explanation
        self.qa_outputs = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 2)
        self.qa_yesno = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 3)

        self._bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self._bert_wordpiece_tokenizer = bert_tokenizer.tokenize
        initializer(self)

        self._official_f1 = Average()
        self._official_EM = Average()

    def bert_offsets_to_wordpiece_offsets(self,bert_offsets):
        # first offset is [CLS]
        wordpiece_offsets = [0]
        last_offset = 0
        for idx, offset in enumerate(bert_offsets):
            wordpiece_offsets += [idx for i in range(last_offset,offset)]
            last_offset = offset
        return wordpiece_offsets

    def _improve_answer_spans(self, span_starts, span_ends, metadata, bert_offsets, bert_wordpiece_ids, mask):

        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.

        # moving to word piece indexes from token indexes of start and end span
        new_span_starts_list = []
        new_span_ends_list = []
        for i, inst_metadata in zip(range(len(metadata)),metadata):
            if span_starts[i] == 0 and span_ends[i] == 0:
                new_span_starts_list.append(0)
                new_span_ends_list.append(0)
            else:

                input_start = bert_offsets[i, span_starts[i]]
                input_end = bert_offsets[i, span_ends[i] + 1] - 1 if bert_offsets[i, span_ends[i] + 1] > 0 else \
                    bert_offsets[i, span_ends[i]]

                # we need to word piece tokenize the passage for this
                word_pieces = [self.vocab.get_token_from_index(int(bert_wordpiece_ids[i,j]), namespace="bert") \
                    for j in range(mask.size(1)) if mask[i,j]]

                tok_answer_text = " ".join(self._bert_wordpiece_tokenizer(inst_metadata['single_answer']))
                found_improved_answer = False
                for new_start in range(input_start, input_end + 1):
                    for new_end in range(input_end, new_start - 1, -1):
                        text_span = " ".join(word_pieces[new_start:(new_end + 1)])
                        if text_span == tok_answer_text:
                            new_span_starts_list.append(new_start)
                            new_span_ends_list.append(new_end)
                            found_improved_answer = True

                if not found_improved_answer:
                    new_span_starts_list.append(input_start)
                    new_span_ends_list.append(input_end)


        return new_span_starts_list, new_span_ends_list


    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_starts: torch.IntTensor = None,
                span_ends: torch.IntTensor = None,
                yesno_labels : torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, num_of_passage_tokens = passage['bert'].size()

        # TODO chaged to ids only
        #embedded_chunk1 = self._text_field_embedder(passage)

        input_ids = passage['bert']
        token_type_ids = torch.zeros_like(input_ids)
        mask = (input_ids != 0).long()
        embedded_chunk, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(token_type_ids),
                                                         attention_mask=util.combine_initial_dims(mask),
                                                         output_all_encoded_layers=False)
        passage_length = embedded_chunk.size(1)

        # BERT for QA is a fully connected linear layer on top of BERT producing 2 vectors of
        # start and end spans.
        logits = self.qa_outputs(embedded_chunk)
        start_logits, end_logits = logits.split(1, dim=-1)
        span_start_logits = start_logits.squeeze(-1)
        span_end_logits = end_logits.squeeze(-1)

        # all input is preprocessed before farword is run, counting the yesno vocabulary
        # will indicate if yesno support is at all needed.
        # TODO add option to override this explicitly
        if self.vocab.get_vocab_size("yesno_labels") > 1:
            yesno_logits = self.qa_yesno(torch.max(embedded_chunk, 1)[0])

        bert_offsets = passage['bert-offsets'].cpu().numpy()
        (wordpiece_span_starts, wordpiece_span_ends) = self._improve_answer_spans(span_starts, span_ends, \
                                                             metadata, bert_offsets, passage['bert'],mask)

        wordpiece_span_starts = torch.cuda.LongTensor(wordpiece_span_starts, device=span_end_logits.device) \
            if torch.cuda.is_available() else torch.LongTensor(wordpiece_span_ends)
        wordpiece_span_ends = torch.cuda.LongTensor(wordpiece_span_starts, device=span_end_logits.device) \
            if torch.cuda.is_available() else torch.LongTensor(wordpiece_span_ends)
        wordpiece_span_starts.clamp_(0, passage_length)
        wordpiece_span_ends.clamp_(0, passage_length)

        loss_fct = CrossEntropyLoss(ignore_index=passage_length)
        start_loss = loss_fct(start_logits.squeeze(-1), wordpiece_span_starts)
        end_loss = loss_fct(end_logits.squeeze(-1), wordpiece_span_ends)

        # Adding some masks with numerically stable values
        if False:
            passage_mask = util.get_text_field_mask(passage).float()

            repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, 1, 1)
            repeated_passage_mask = repeated_passage_mask.view(batch_size, passage_length)
            span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
            span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)

            inds_with_gold_answer = np.argwhere(span_starts.view(-1).cpu().numpy() >= 0)
            inds_with_gold_answer = inds_with_gold_answer.squeeze() if len(inds_with_gold_answer) > 1 else inds_with_gold_answer

            loss = 0
            if len(inds_with_gold_answer) > 0:
                loss += nll_loss(util.masked_log_softmax(span_start_logits[inds_with_gold_answer], \
                                                        repeated_passage_mask[inds_with_gold_answer]), \
                                span_starts.view(-1)[inds_with_gold_answer], ignore_index=-1)
                loss += nll_loss(util.masked_log_softmax(span_end_logits[inds_with_gold_answer], \
                                                         repeated_passage_mask[inds_with_gold_answer]), \
                                 span_ends.view(-1)[inds_with_gold_answer], ignore_index=-1)

        if self.vocab.get_vocab_size("yesno_labels") > 1 and yesno_labels is not None:
            yesno_loss = loss_fct(yesno_logits, yesno_labels)
            loss = (start_loss + end_loss + yesno_loss) / 3
        else:
            loss = (start_loss + end_loss) / 2

        output_dict: Dict[str, Any] = {}
        if loss == 0:
            # For evaluation purposes only!
            output_dict["loss"] = torch.cuda.FloatTensor([0], device=span_end_logits.device) \
                if torch.cuda.is_available() else torch.FloatTensor([0])
        else:
            output_dict["loss"] = loss

        # Compute F1 and preparing the output dictionary.
        output_dict['best_span_str'] = []
        output_dict['best_span_logit'] = []
        output_dict['yesno'] = []
        output_dict['yesno_logit'] = []
        output_dict['qid'] = []
        if span_starts is not None:
            output_dict['EM'] = []
            output_dict['f1'] = []


        # getting best span prediction for
        best_span = self._get_example_predications(span_start_logits, span_end_logits, self._max_span_length)
        best_span_cpu = best_span.detach().cpu().numpy()

        for instance_ind, instance_metadata in zip(range(batch_size), metadata):
            best_span_logit = span_start_logits.data.cpu().numpy()[instance_ind, best_span_cpu[instance_ind][0]] + \
                              span_end_logits.data.cpu().numpy()[instance_ind, best_span_cpu[instance_ind][1]]

            if self.vocab.get_vocab_size("yesno_labels") > 1:
                yesno_maxind = np.argmax(yesno_logits[instance_ind].data.cpu().numpy())
                yesno_logit = yesno_logits[instance_ind, yesno_maxind].data.cpu().numpy()
                yesno_pred = self.vocab.get_token_from_index(yesno_maxind, namespace="yesno_labels")
            else:
                yesno_pred = 'no_yesno'
                yesno_logit = -30.0

            passage_str = instance_metadata['original_passage']
            offsets = instance_metadata['token_offsets']

            predicted_span = best_span_cpu[instance_ind]
            # In this version yesno if not "no_yesno" will be regarded as final answer before the spans are considered.
            if yesno_pred != 'no_yesno':
                best_span_string = yesno_pred
            else:
                if predicted_span[0] == 0 and predicted_span[1] == 0:
                    best_span_string = 'cannot_answer'
                else:
                    wordpiece_offsets = self.bert_offsets_to_wordpiece_offsets(bert_offsets[instance_ind][0:len(offsets)])
                    start_offset = offsets[wordpiece_offsets[predicted_span[0] if predicted_span[0] < len(wordpiece_offsets) \
                        else len(wordpiece_offsets)-1]][0]
                    end_offset = offsets[wordpiece_offsets[predicted_span[1] if predicted_span[1] < len(wordpiece_offsets) \
                        else len(wordpiece_offsets)-1]][1]
                    best_span_string = passage_str[start_offset:end_offset]

            output_dict['best_span_str'].append(best_span_string)
            output_dict['best_span_logit'].append(best_span_logit)
            output_dict['yesno'].append(yesno_pred)
            output_dict['yesno_logit'].append(yesno_logit)
            output_dict['qid'].append(instance_metadata['question_id'])

            # In prediction mode we have no gold answers
            if span_starts is not None:
                yesno_label_ind = yesno_labels.data.cpu().numpy()[instance_ind]
                yesno_label = self.vocab.get_token_from_index(yesno_label_ind, namespace="yesno_labels")

                if yesno_label != 'no_yesno':
                    gold_answer_texts = [yesno_label]
                elif instance_metadata['cannot_answer']:
                    gold_answer_texts = ['cannot_answer']
                else:
                    gold_answer_texts = instance_metadata['answer_texts_list']

                f1_score = squad_eval.metric_max_over_ground_truths(squad_eval.f1_score, best_span_string, gold_answer_texts)
                EM_score = squad_eval.metric_max_over_ground_truths(squad_eval.exact_match_score, best_span_string, gold_answer_texts)
                self._official_f1(100 * f1_score)
                self._official_EM(100 * EM_score)
                output_dict['EM'].append(100 * EM_score)
                output_dict['f1'].append(100 * f1_score)



        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'EM': self._official_EM.get_metric(reset),
                'f1': self._official_f1.get_metric(reset)}


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

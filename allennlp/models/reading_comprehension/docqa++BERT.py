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
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.nn import InitializerApplicator, util
from allennlp.tools import squad_eval
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# ALON - for line profiler
try:
    profile
except NameError:
    profile = lambda x: x

@Model.register("docqa++BERT")
class DocQAPlusBERT(Model):
    """
    This class implements modified version of BiDAF
    (with self attention and residual layer, from Clark and Gardner ACL 17 paper) model as used in
    Question Answering in Context (EMNLP 2018) paper [https://arxiv.org/pdf/1808.07036.pdf].

    In this set-up, a single instance is a dialog, list of question answer pairs.

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
    multi_choice_answers: ``bool``,optional (default=False)
        If True, dataset is multi-choice answer, and accuracy will be computed accurdigly.
        Note that "multichoice_incorrect_answers" must be provided in the dataset.
    num_context_answers : ``int``, optional (default=0)
        If greater than 0, the model will consider previous question answering context.
    max_span_length: ``int``, optional (default=0)
        Maximum token length of the output span.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 multi_choice_answers: int = 0,
                 frac_of_validation_used: float = 1.0,
                 frac_of_training_used: float = 1.0,
                 shared_norm: bool = True,
                 support_yesno: bool = False,
                 support_followup: bool = False,
                 num_context_answers: int = 0,
                 max_qad_triplets: int = 0,
                 max_span_length: int = 30,
                 predictions_file = None,
                 use_multi_label_loss: bool = False,
                 stats_report_freq:float = None,
                 debug_experiment_name:str = None) -> None:
        super().__init__(vocab)
        self._num_context_answers = num_context_answers
        self._multi_choice_answers = multi_choice_answers
        self._support_yesno = support_yesno
        self._support_followup = support_followup
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._shared_norm = shared_norm
        self._stats_report_freq = stats_report_freq
        self._debug_experiment_name = debug_experiment_name
        self._use_multi_label_loss = use_multi_label_loss
        self._predictions_file = predictions_file

        if predictions_file is not None and os.path.isfile(predictions_file):
            os.remove(predictions_file)


        # see usage below for explanation
        self._all_qa_count = 0
        self._qas_used_fraction = 1.0
        self._max_qad_triplets = max_qad_triplets
        self._frac_of_validation_used = frac_of_validation_used
        self._frac_of_training_used = frac_of_training_used

        self.qa_outputs = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 2)

        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        if self._multi_choice_answers:
            self._multichoice_accuracy = BooleanAccuracy()
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

    def shared_norm_cross_entropy_loss(self, span_logits, answers, passage_length):
        target = torch.cuda.FloatTensor(1, passage_length * answers.size(0), device=span_logits.device) \
            if torch.cuda.is_available() else torch.FloatTensor(1, passage_length * answers.size(0))
        target.zero_()

        answers = answers.squeeze().cpu() if len(answers) > 1 else answers.cpu()

        for ind, q_target in enumerate(answers):
            if len(np.argwhere(q_target.squeeze() >= 0))>0 :
                target[0, q_target[(q_target >= 0) & (q_target < passage_length)] + passage_length * ind] = 1.0

        return -(torch.log((F.softmax(torch.cat(tuple(span_logits)), dim=-1) *  target.float()).sum(dim=1))).mean()

    @profile
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
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
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of the followings.
        Each of the followings is a nested list because first iterates over dialog, then questions in dialog.

        qid : List[List[str]]
            A list of list, consisting of question ids.
        best_span_str : List[List[str]]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        batch_size, num_of_passage_tokens = passage['bert'].size()
        #if passage['bert'].size(1) < 384:
        #    passage['bert'] = torch.nn.functional.pad(passage['bert'], (0, 384 - passage['bert'].size(1)), "constant", 0)
        #if passage['bert-offsets'].size(1) < 384:
        #    passage['bert-offsets'] = torch.nn.functional.pad(passage['bert-offsets'], (0, 384 - passage['bert-offsets'].size(1)), "constant", 0)
        #if passage['mask'].size(1) < 384:
        #    passage['mask'] = torch.nn.functional.pad(passage['mask'], (0, 384 - passage['mask'].size(1)), "constant", 1)

        if random.randint(1, 100) % 100 == 0:
            for meta in metadata:
                logger.info("%s %s", meta['dataset'], meta['instance_id'])

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

        # Fraction of Examples Used. (for True accuracy calculations)
        # NOTE (TODO) this is a workaround, we cannot save global information to be passed to the model yet
        # (see https://github.com/allenai/allennlp/issues/1809) so we will save it every time it changes
        # insuring that if we do a full pass on the validation set and take max for all_qa_count we will
        # get the correct number (except if the last ones are skipped.... hopefully this is a small diff )

        intances_question_id = [insta_meta['question_id'] for insta_meta in metadata]
        question_instances_split_inds = np.cumsum(np.unique(intances_question_id, return_counts=True)[1])[:-1]
        per_question_inds = np.split(range(batch_size), question_instances_split_inds)
        metadata = np.split(metadata, question_instances_split_inds)
        self._qas_used_fraction = metadata[0][0]['qas_used_fraction']

        # Compute the loss.
        if span_start is not None and len(np.argwhere(span_start.squeeze().cpu() >= 0)) > 0:
            if self._shared_norm:
                loss = 0
                loss_steps = 0

                # For every context/question
                for question_inds, metadata_list in zip(per_question_inds,metadata):

                    # Could of wrote this shorter but it's clearer like this ...
                    if len(question_inds)==0:
                        continue

                    inds_with_gold_answer = np.argwhere(span_start.view(-1)[question_inds].cpu().numpy() >= 0)
                    inds_with_gold_answer = inds_with_gold_answer.squeeze() if len(
                        inds_with_gold_answer) > 1 else inds_with_gold_answer

                    if len(inds_with_gold_answer)==0:
                        continue

                    if self._use_multi_label_loss:
                        try:
                            loss += self.shared_norm_cross_entropy_loss(span_start_logits[question_inds], \
                                                                        span_start[question_inds], passage_length)
                            loss += self.shared_norm_cross_entropy_loss(span_end_logits[question_inds], \
                                                                        span_end[question_inds], passage_length)
                        except:
                            ElasticLogger().write_log('INFO', 'Loss Error', \
                                                      context_dict={'span_start_logits': span_start_logits[question_inds].cpu().size(),
                                                                      'span_end_logits_size': span_end_logits[question_inds].cpu().size(),
                                                                      'span_start': span_start[question_inds].squeeze().cpu().numpy().tolist(),
                                                                      'span_end': span_end[question_inds].squeeze().cpu().numpy().tolist(),
                                                                      'error_message': traceback.format_exc(),
                                                                      'batch_size': batch_size,
                                                                      'passage_length': passage_length}, print_log=True)
                            a = torch.autograd.Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
                            loss = torch.sum(a ** 2)
                            output_dict["loss"] = loss
                    else:

                        span_start_logits_softmaxed = util.masked_log_softmax(\
                            torch.cat(tuple(span_start_logits[question_inds])).unsqueeze(0), \
                            torch.cat(tuple(repeated_passage_mask[question_inds])).unsqueeze(0))
                        span_end_logits_softmaxed = util.masked_log_softmax(
                            torch.cat(tuple(span_end_logits[question_inds])).unsqueeze(0), \
                            torch.cat(tuple(repeated_passage_mask[question_inds])).unsqueeze(0))


                        span_start_logits_softmaxed = span_start_logits_softmaxed.reshape(len(question_inds),span_start_logits.size(1))
                        span_end_logits_softmaxed = span_end_logits_softmaxed.reshape(len(question_inds), span_start_logits.size(1))

                        # computing loss only for indexes with answers
                        loss += nll_loss(span_start_logits_softmaxed[inds_with_gold_answer], \
                                         span_start.view(-1)[question_inds[inds_with_gold_answer]], ignore_index=-1)
                        loss += nll_loss(span_end_logits_softmaxed[inds_with_gold_answer], \
                                         span_end.view(-1)[question_inds[inds_with_gold_answer]], ignore_index=-1)
                    loss_steps += 1

                if loss_steps > 0:
                    loss /= loss_steps
                    output_dict["loss"] = loss
            else:
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


            # TODO these are not updates
            #self._span_start_accuracy(span_start_logits, span_start.view(-1))
            #self._span_end_accuracy(span_end_logits, span_end.view(-1))
            #self._span_accuracy(best_span[:, 0:2],torch.stack([span_start, span_end], -1).view(total_qa_count, 2))
        # TODO: This is a patch, for dev question with no answer token found,
        # but we would like to see if we still get F1 score for it...
        # so in evaluation our loss is not Accurate! (however the question with no answer tokens will
        # remain the same number so it is relatively accuracy)
        if not self.training and 'loss' not in output_dict:
            output_dict["loss"] = torch.cuda.FloatTensor([0], device=span_end_logits.device) \
                if torch.cuda.is_available() else torch.FloatTensor([0])

        # support for multi choice answers:
        # TODO this does not handle prediction mode at all .....
        # we iterate over document that do not contain the golden answer for validation and test setup.
        span_start_logits_numpy = span_start_logits.data.cpu().numpy()
        span_end_logits_numpy = span_end_logits.data.cpu().numpy()


        # Compute F1 and preparing the output dictionary.
        output_dict['best_span_str'] = []
        output_dict['qid'] = []

        ## TODO UGLY PATCH FOR TESTING
        #new_metadata = []
        #for question_meta in metadata:
        #    new_metadata += [question_meta for i in range(num_of_docs)]
        #metadata = new_metadata

        # best_span is a vector of more than one span
        best_span_cpu = best_span.detach().cpu().numpy()

        # Iterating over every question (which may contain multiple instances, one per document)

        for question_inds, question_instances_metadata in zip(per_question_inds, metadata):
            if len(question_inds) == 0:
                continue

            # We need to perform softmax here !!
            best_span_ind = np.argmax(span_start_logits_numpy[question_inds, best_span_cpu[question_inds][:, 0]] +
                      span_end_logits_numpy[question_inds, best_span_cpu[question_inds][:, 1]])
            best_span_logit = np.max(span_start_logits_numpy[question_inds, best_span_cpu[question_inds][:, 0]] +
                                      span_end_logits_numpy[question_inds, best_span_cpu[question_inds][:, 1]])

            # TODO this shouldent happen - we should consider spans from passages not taken...
            #if span_start.view(-1)[question_inds[best_span_ind]] == -1:
            #    self._official_f1(100 * 0.0)
            #    self._official_EM(100 * 0.0)
            #    continue

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
                if len(gold_answer_texts) > 1:
                    t_f1 = []
                    t_EM = []
                    for answer_index in range(len(gold_answer_texts)):
                        idxes = list(range(len(gold_answer_texts)))

                        refs = [gold_answer_texts[z] for z in idxes]
                        t_f1.append(squad_eval.metric_max_over_ground_truths(squad_eval.f1_score,best_span_string,refs))
                        t_EM.append(squad_eval.metric_max_over_ground_truths(squad_eval.exact_match_score,best_span_string,refs))
                    f1_score = 1.0 * sum(t_f1) / len(t_f1)
                    EM_score = 1.0 * sum(t_EM) / len(t_EM)
                else:
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
                                'gold_answer_texts':gold_answer_texts}) + '\n')
        #output_dict['qid'].append(per_dialog_query_id_list)
        #output_dict['best_span_str'].append(per_dialog_best_span_list)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        yesno_tags = [[self.vocab.get_token_from_index(x, namespace="yesno_labels") for x in yn_list] \
                      for yn_list in output_dict.pop("yesno")]
        followup_tags = [[self.vocab.get_token_from_index(x, namespace="followup_labels") for x in followup_list] \
                         for followup_list in output_dict.pop("followup")]
        output_dict['yesno'] = yesno_tags
        output_dict['followup'] = followup_tags
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        # calculating final accuracy considering fraction of examples used in dataset creation, and
        # number used after data-reader filter namely "self._examples_used_frac"
        frac_used = self._qas_used_fraction

        return {'EM': self._official_EM.get_metric(reset) * frac_used,
                'f1': self._official_f1.get_metric(reset) * frac_used,
                'qas_used_fraction': frac_used}


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

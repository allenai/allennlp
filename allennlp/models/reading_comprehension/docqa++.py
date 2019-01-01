import logging
from allennlp.common.elastic_logger import ElasticLogger
from typing import Any, Dict, List
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
import inspect

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


@Model.register("docqa++")
class BidafPlusPlus(Model):
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
                 phrase_layer: Seq2SeqEncoder,
                 residual_encoder: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
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
                 marker_embedding_dim: int = 10,
                 max_span_length: int = 30) -> None:
        super().__init__(vocab)
        self._num_context_answers = num_context_answers
        self._multi_choice_answers = multi_choice_answers
        self._support_yesno = support_yesno
        self._support_followup = support_followup
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._shared_norm = shared_norm
        self._phrase_layer = phrase_layer
        self._marker_embedding_dim = marker_embedding_dim
        self._encoding_dim = phrase_layer.get_output_dim()
        max_turn_length = 12

        # see usage below for explanation
        self._all_qa_count = 0
        self._examples_used_frac = 1.0
        self._max_qad_triplets = max_qad_triplets
        self._frac_of_validation_used = frac_of_validation_used
        self._frac_of_training_used = frac_of_training_used

        self._matrix_attention = LinearMatrixAttention(self._encoding_dim, self._encoding_dim, 'x,y,x*y')
        self._merge_atten = TimeDistributed(torch.nn.Linear(self._encoding_dim * 4, self._encoding_dim))

        self._residual_encoder = residual_encoder

        if num_context_answers > 0:
            self._question_num_marker = torch.nn.Embedding(max_turn_length,
                                                           marker_embedding_dim * num_context_answers)
            self._prev_ans_marker = torch.nn.Embedding((num_context_answers * 4) + 1, marker_embedding_dim)

        self._self_attention = LinearMatrixAttention(self._encoding_dim, self._encoding_dim, 'x,y,x*y')

        self._followup_lin = torch.nn.Linear(self._encoding_dim, 3)
        self._merge_self_attention = TimeDistributed(torch.nn.Linear(self._encoding_dim * 3,
                                                                     self._encoding_dim))

        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder

        self._span_start_predictor = TimeDistributed(torch.nn.Linear(self._encoding_dim, 1))
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(self._encoding_dim, 1))
        if self._support_yesno:
            self._span_yesno_predictor = TimeDistributed(torch.nn.Linear(self._encoding_dim, 3))
        self._span_followup_predictor = TimeDistributed(self._followup_lin)

        check_dimensions_match(phrase_layer.get_input_dim(),
                               text_field_embedder.get_output_dim() +
                               marker_embedding_dim * num_context_answers,
                               "phrase layer input dim",
                               "embedding dim + marker dim * num context answers")

        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        if self._support_yesno:
            self._span_yesno_accuracy = CategoricalAccuracy()
        if self._support_followup:
            self._span_followup_accuracy = CategoricalAccuracy()
        if self._support_yesno:
            self._span_gt_yesno_accuracy = CategoricalAccuracy()
        if self._support_followup:
            self._span_gt_followup_accuracy = CategoricalAccuracy()

        self._span_accuracy = BooleanAccuracy()
        if self._multi_choice_answers:
            self._multichoice_accuracy = BooleanAccuracy()
        self._official_f1 = Average()
        self._official_EM = Average()
        self._variational_dropout = InputVariationalDropout(dropout)

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
        p1_answer_marker : ``torch.IntTensor``, optional
            This is one of the inputs, but only when num_context_answers > 0.
            This is a tensor that has a shape [batch_size, max_qa_count, max_passage_length].
            Most passage token will have assigned 'O', except the passage tokens belongs to the previous answer
            in the dialog, which will be assigned labels such as <1_start>, <1_in>, <1_end>.
            For more details, look into dataset_readers/util/make_reading_comprehension_instance_quac
        p2_answer_marker :  ``torch.IntTensor``, optional
            This is one of the inputs, but only when num_context_answers > 1.
            It is similar to p1_answer_marker, but marking previous previous answer in passage.
        p3_answer_marker :  ``torch.IntTensor``, optional
            This is one of the inputs, but only when num_context_answers > 2.
            It is similar to p1_answer_marker, but marking previous previous previous answer in passage.
        yesno_list :  ``torch.IntTensor``, optional
            This is one of the outputs that we are trying to predict.
            Three way classification (the yes/no/not a yes no question).
        followup_list :  ``torch.IntTensor``, optional
            This is one of the outputs that we are trying to predict.
            Three way classification (followup / maybe followup / don't followup).
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
        followup : List[List[int]]
            A list of list, consisting of continuation marker prediction index.
            (y :yes, m: maybe follow up, n: don't follow up)
        yesno : List[List[int]]
            A list of list, consisting of affirmation marker prediction index.
            (y :yes, x: not a yes/no question, n: np)
        best_span_str : List[List[str]]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # TODO this repeat is ugly ... (simulating 1 questions.. )
        for key in question.keys():
            question[key] = question[key].unsqueeze(1)

        batch_size, max_qa_count, max_q_len, _ = question['token_characters'].size()
        total_qa_count = batch_size * max_qa_count

        # NOTE we assume that the batch instances are sorted per question!
        # in document qa setup we usually use only training triplets (question, answer ,context) that
        # contain the golden answer, to save tranining time.
        intances_question_id = [insta_meta['question_id'] for insta_meta in metadata]
        question_instances_split_inds = np.cumsum(np.unique(intances_question_id, return_counts=True)[1])[:-1]
        per_question_inds = np.split(range(total_qa_count), question_instances_split_inds)
        metadata = np.split(metadata, question_instances_split_inds)

        ## TODO this should be in iterator
        #if self._max_qad_triplets > 0:
        #    if instaces_with_golden_answer.size >= self._max_qad_triplets:
        #        instaces_with_golden_answer = instaces_with_golden_answer[0:self._max_qad_triplets]

        # Todo filtering instances with golden answer should be done in iterator
        if False:
            if self.training:
                for type in question.keys():
                    question[type] = question[type][instaces_with_golden_answer]
                for type in passage.keys():
                    passage[type] = passage[type][instaces_with_golden_answer]

                total_qa_count = len(instaces_with_golden_answer)

                selected_span_start = span_start.view(-1)[instaces_with_golden_answer]
                selected_span_end = span_end.view(-1)[instaces_with_golden_answer]
            else:
                selected_span_start = span_start.view(-1)
                selected_span_end = span_end.view(-1)

            # building mappings between instances and (q,a,d) triplets
            golden_answer_instance_triplets = []
            golden_answer_instance_offset = []
            golden_answer_offset = 0
            for batch_ind, inst_metadata in enumerate(metadata):
                golden_answer_instance_triplets.append([])
                golden_answer_instance_offset.append([])
                for instance_offset, ind in enumerate(range(batch_ind * num_of_docs, (batch_ind + 1) * num_of_docs)):
                    if self.training:
                        if ind in instaces_with_golden_answer:
                            golden_answer_instance_triplets[batch_ind].append(golden_answer_offset)
                            golden_answer_offset += 1
                            golden_answer_instance_offset[batch_ind].append(instance_offset)
                    else:
                        golden_answer_instance_triplets[batch_ind].append(ind)
                        golden_answer_instance_offset[batch_ind].append(instance_offset)


        # questions embedding
        embedded_question = self._text_field_embedder(question, num_wrapping_dims=1)
        embedded_question = embedded_question.reshape(total_qa_count, max_q_len,
                                                      self._text_field_embedder.get_output_dim())
        embedded_question = self._variational_dropout(embedded_question)

        # context embedding
        embedded_passage = self._variational_dropout(self._text_field_embedder(passage))
        passage_length = embedded_passage.size(1)

        # context repeating (as the amount of qas)
        question_mask = util.get_text_field_mask(question, num_wrapping_dims=1).float()
        question_mask = question_mask.reshape(total_qa_count, max_q_len)
        passage_mask = util.get_text_field_mask(passage).float()

        repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, max_qa_count, 1)
        repeated_passage_mask = repeated_passage_mask.view(total_qa_count, passage_length)

        encoded_passage = self._variational_dropout(self._phrase_layer(embedded_passage, passage_mask))
        repeated_encoded_passage = encoded_passage.unsqueeze(1).repeat(1, max_qa_count, 1, 1)
        repeated_encoded_passage = repeated_encoded_passage.view(total_qa_count,
                                                                 passage_length,
                                                                 self._encoding_dim)

        encoded_question = self._variational_dropout(self._phrase_layer(embedded_question, question_mask))

        # Shape: (batch_size * max_qa_count, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(repeated_encoded_passage, encoded_question)
        # Shape: (batch_size * max_qa_count, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size * max_qa_count, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)

        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        question_passage_attention = util.masked_softmax(question_passage_similarity, repeated_passage_mask)
        # Shape: (batch_size * max_qa_count, encoding_dim)
        question_passage_vector = util.weighted_sum(repeated_encoded_passage, question_passage_attention)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(total_qa_count,
                                                                                    passage_length,
                                                                                    self._encoding_dim)

        # Shape: (batch_size * max_qa_count, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([repeated_encoded_passage,
                                          passage_question_vectors,
                                          repeated_encoded_passage * passage_question_vectors,
                                          repeated_encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        final_merged_passage = F.relu(self._merge_atten(final_merged_passage))

        residual_layer = self._variational_dropout(self._residual_encoder(final_merged_passage,
                                                                          repeated_passage_mask))
        self_attention_matrix = self._self_attention(residual_layer, residual_layer)

        mask = repeated_passage_mask.reshape(total_qa_count, passage_length, 1) \
               * repeated_passage_mask.reshape(total_qa_count, 1, passage_length)
        self_mask = torch.eye(passage_length, passage_length, device=self_attention_matrix.device)
        self_mask = self_mask.reshape(1, passage_length, passage_length)
        mask = mask * (1 - self_mask)

        self_attention_probs = util.masked_softmax(self_attention_matrix, mask)

        # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
        self_attention_vecs = torch.matmul(self_attention_probs, residual_layer)
        self_attention_vecs = torch.cat([self_attention_vecs, residual_layer,
                                         residual_layer * self_attention_vecs],
                                        dim=-1)
        residual_layer = F.relu(self._merge_self_attention(self_attention_vecs))

        final_merged_passage = final_merged_passage + residual_layer
        # batch_size * maxqa_pair_len * max_passage_len * 200
        final_merged_passage = self._variational_dropout(final_merged_passage)
        start_rep = self._span_start_encoder(final_merged_passage, repeated_passage_mask)
        span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)

        end_rep = self._span_end_encoder(torch.cat([final_merged_passage, start_rep], dim=-1),
                                         repeated_passage_mask)
        span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)


        span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
        # batch_size * maxqa_len_pair, max_document_len
        span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)

        best_span = self._get_example_predications(span_start_logits, span_end_logits,self._max_span_length)

        output_dict: Dict[str, Any] = {}

        # Fraction of Examples Used. (for True accuracy calculations)
        # NOTE (TODO) this is a workaround, we cannot save global information to be passed to the model yet
        # (see https://github.com/allenai/allennlp/issues/1809) so we will save it every time it changes
        # insuring that if we do a full pass on the validation set and take max for all_qa_count we will
        # get the correct number (except if the last ones are skipped.... hopefully this is a small diff )
        
        self._all_qa_count = metadata[0][0]['num_examples_used'][1]
        self._examples_used_frac = float(metadata[0][0]['num_examples_used'][0]) / metadata[0][0]['num_examples_used'][1]
        print(self._examples_used_frac)


        # Compute the loss.
        if span_start is not None:
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


                    # TODO filtering result with no golden answer for loss, should we not compute this at all to save time?

                    span_start_logits_softmaxed = util.masked_log_softmax(\
                        torch.cat(tuple(span_start_logits[question_inds])).unsqueeze(0), \
                        torch.cat(tuple(repeated_passage_mask[question_inds])).unsqueeze(0))
                    span_end_logits_softmaxed = util.masked_log_softmax(
                        torch.cat(tuple(span_end_logits[question_inds])).unsqueeze(0), \
                        torch.cat(tuple(repeated_passage_mask[question_inds])).unsqueeze(0))

                    ## Log then Sum for share norm implementation
                    #span_start_logits_softmaxed = util.masked_softmax( \
                    #    torch.cat(tuple(span_start_logits[question_inds])).unsqueeze(0), \
                    #    torch.cat(tuple(repeated_passage_mask[question_inds])).unsqueeze(0))
                    #span_end_logits_softmaxed = util.masked_softmax(
                    #    torch.cat(tuple(span_end_logits[question_inds])).unsqueeze(0), \
                    #    torch.cat(tuple(repeated_passage_mask[question_inds])).unsqueeze(0))
                    #start_indexes = [ind + doc_num * passage_length for doc_num, ind in
                    #         enumerate(selected_span_start[question_inds])]
                    #end_indexes = [ind + doc_num * passage_length for doc_num, ind in
                    #         enumerate(selected_span_end[question_inds])]
                    #dummy_target = torch.cuda.LongTensor([0],device=span_start_logits_softmaxed.device) \
                    #    if torch.cuda.is_available() else torch.LongTensor([0])
                    #loss += nll_loss(torch.log(torch.sum(span_start_logits_softmaxed[0,start_indexes])).unsqueeze(0).unsqueeze(0), \
                    #                 dummy_target, ignore_index=-1)
                    #loss += nll_loss(torch.log(torch.sum(span_end_logits_softmaxed[0, end_indexes])).unsqueeze(0).unsqueeze(0), \
                    #                 dummy_target, ignore_index=-1)

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
                inds_with_gold_answer = np.argwhere(span_start.view(-1).cpu().numpy() >= 0)
                inds_with_gold_answer = inds_with_gold_answer.squeeze() if len(inds_with_gold_answer) > 1 else inds_with_gold_answer

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

            # support for multi choice answers:
            # TODO this does not handle prediction mode at all .....
            # we iterate over document that do not contain the golden answer for validation and test setup.
            span_start_logits_numpy = span_start_logits.data.cpu().numpy()
            span_end_logits_numpy = span_end_logits.data.cpu().numpy()

            if False and self._multi_choice_answers:

                for batch_ind,inst_metadata in enumerate(metadata):
                    max_correct_answer = -50
                    max_incorrect_answer = -50

                    if self.training:
                        instance_triplets = golden_answer_instance_triplets[batch_ind]
                    else:
                        instance_triplets = range(batch_ind * num_of_docs, (batch_ind + 1) * num_of_docs)

                    if len(instance_triplets) == 0:
                        continue

                    #print('---------------------------')
                    #print(inst_metadata['question'])
                    # computing the max score of the correct answer
                    for offest,j in zip(golden_answer_instance_offset[batch_ind],range(len(instance_triplets))):

                        if offest < len(inst_metadata["token_span_lists"]['answers']):
                            for answer_start_end in inst_metadata['token_span_lists']['answers'][offest][0]:
                                #print(inst_metadata['passage_tokens'][offest][answer_start_end[0]:answer_start_end[1]+1])
                                score = span_start_logits_numpy[instance_triplets[j]][answer_start_end[0]] \
                                        + span_end_logits_numpy[instance_triplets[j]][answer_start_end[1]]
                                if score>max_correct_answer:
                                    max_correct_answer = score

                        # computing the max score of the incorrect answers
                        if offest < len(inst_metadata["token_span_lists"]['distractor_answers']):
                            for answer_start_end in inst_metadata['token_span_lists']['distractor_answers'][offest][0]:
                                #print(inst_metadata['passage_tokens'][offest][answer_start_end[0]:answer_start_end[1]+1])
                                score = span_start_logits_numpy[instance_triplets[j]][answer_start_end[0]] \
                                        + span_end_logits_numpy[instance_triplets[j]][answer_start_end[1]]
                                if score > max_incorrect_answer:
                                    max_incorrect_answer = score

                    # If max currect answer score is higher, then multi_choice accuracy bool accuracy is True.
                    self._multichoice_accuracy(torch.Tensor([(max_correct_answer > max_incorrect_answer) * 1]),torch.Tensor([1]))

            

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
        if self.training:
            frac_used = self._examples_used_frac * self._frac_of_training_used
        else:
            frac_used = self._examples_used_frac * self._frac_of_validation_used
        print(frac_used, self._examples_used_frac, self._frac_of_training_used, self._frac_of_training_used)
        
        return {'EM': self._official_EM.get_metric(reset) * frac_used,
                'f1': self._official_f1.get_metric(reset) * frac_used,
                'examples_used_frac': frac_used}


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

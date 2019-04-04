import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random
import numpy as np

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# ALON - for line profiler
try:
    profile
except NameError:
    profile = lambda x: x

def sample_contexts(instance_list,sample_size):
    random.seed(2)

    instance_list = sorted(instance_list, key=lambda x: x['metadata']['question_id'])
    intances_question_id = [instance['metadata']['question_id'] for instance in instance_list]
    split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
    per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                              range(len(split_inds) - 1)]

    random.shuffle(per_question_instances)

    sampled_contexts = []
    num_of_qas = 0
    for question_instances in per_question_instances:
        if num_of_qas >= sample_size:
            break
        sampled_contexts += question_instances
        num_of_qas += 1
    return sampled_contexts





@DatasetReader.register("multiqa_bert_mix_mrqa")
class BERTQAReaderMixMRQA(DatasetReader):
    """
    Reads a JSON-formatted Quesiton Answering in Context (QuAC) data file
    and returns a ``Dataset`` where the ``Instances`` have four fields: ``question``, a ``ListField``,
    ``passage``, another ``TextField``, and ``span_start`` and ``span_end``, both ``ListField`` composed of
    IndexFields`` into the ``passage`` ``TextField``.
    Two ``ListField``, composed of ``LabelField``, ``yesno_list`` and  ``followup_list`` is added.
    We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_text_lists'] and ``metadata['token_offsets']``.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_one_inst_per_question = False,
                 rewind_datasets: bool = False,
                 sampling_ratio = None,
                 lazy: bool = False,
                 all_question_instances_in_batch = False,
                 sample_size: int = -1,
                 ) -> None:
        super().__init__(lazy)
        random.seed(0)
        logger.info('----------------- NEW SEED ---------------')
        self._tokenizer = tokenizer or WordTokenizer()
        self._rewind_datasets = rewind_datasets
        self._sampling_ratio = sampling_ratio
        self._sample_size = sample_size
        self._use_one_inst_per_question = use_one_inst_per_question
        self._all_question_instances_in_batch = all_question_instances_in_batch
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self._bert_wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer.tokenize
        self._never_lowercase = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

    def tokens_to_wordpieces(self, tokens, _bert_do_lowercase=True):
        total_len = 0
        all_word_pieces = []
        for token in tokens:
            text = (token[0].lower()
                    if _bert_do_lowercase and token[0] not in self._never_lowercase
                    else token[0])

            word_pieces = self._bert_wordpiece_tokenizer(text)
            all_word_pieces += word_pieces
            total_len += len(word_pieces)
        return all_word_pieces, total_len

    def make_instances(self, unproc_context, header, _bert_do_lowercase=True):
        STRIDE = 128
        MAX_WORDPIECES = 510
        # TODO remember to convert all separators to [SEP]
        # converting the context into word pieces

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of 128.
        # splitting into chuncks that are not larger than 510 token pieces (NOTE AllenNLP
        # adds the [CLS] and [SEP] token pieces automatically
        instances = []
        for qa in unproc_context['qas']:
            curr_token_ix = 0
            window_start_token_offset = 0
            while curr_token_ix < len(unproc_context['context_tokens']):
                _, num_of_question_wordpieces = self.tokens_to_wordpieces(qa['question_tokens'])
                curr_token_ix = window_start_token_offset

                curr_context_tokens = qa['question_tokens'] + [['[SEP]',len(qa['question']) + 1]]
                context_char_offset = unproc_context['context_tokens'][curr_token_ix][1]
                question_char_offset = len(qa['question']) + 5 + 1 + 1
                num_of_wordpieces = 0
                while num_of_wordpieces < MAX_WORDPIECES - num_of_question_wordpieces - 1 \
                        and curr_token_ix < len(unproc_context['context_tokens']):
                    curr_token = copy.deepcopy(unproc_context['context_tokens'][curr_token_ix])

                    # BERT has only [SEP] in it's word piece vocabulary. because we keps all separators char length 5
                    # we can replace all of them with [SEP] without modifying the offset
                    if curr_token[0] in ['[TLE]','[PAR]','[DOC]']:
                        curr_token[0] = '[SEP]'

                    # fixing the car offset of each token
                    curr_token[1] += question_char_offset - context_char_offset

                    text = (curr_token[0].lower()
                            if _bert_do_lowercase and curr_token[0] not in self._never_lowercase
                            else curr_token[0])

                    word_pieces = self._bert_wordpiece_tokenizer(text)
                    num_of_wordpieces += len(word_pieces)
                    if num_of_wordpieces < MAX_WORDPIECES - num_of_question_wordpieces - 1:
                        window_end_token_offset = curr_token_ix + 1
                        curr_context_tokens.append(curr_token)
                    curr_token_ix += 1


                inst = {}
                inst['question_tokens'] = qa['question_tokens']
                inst['tokens'] = curr_context_tokens
                inst['text'] = qa['question'] + ' [SEP] ' + unproc_context['context'][context_char_offset: \
                            context_char_offset + curr_context_tokens[-1][1] + len(curr_context_tokens[-1][0]) + 1]
                inst['answers'] = []
                qa_metadata = {'has_answer': False, 'dataset': header['dataset'], "question_id": qa['qid'], \
                               'answer_texts_list': list(set(qa['answers']))}
                for answer in qa['detected_answers']:
                    # TODO assuming only one instance per answer

                    if answer['token_spans'][0][0] >= window_start_token_offset and \
                        answer['token_spans'][0][1] < window_end_token_offset:
                        qa_metadata['has_answer'] = True
                        answer_token_offset = len(qa['question_tokens']) + 1 - window_start_token_offset
                        inst['answers'].append((answer['token_spans'][0][0] + answer_token_offset, \
                                                       answer['token_spans'][0][1] + answer_token_offset,
                                                       answer['text']))

                        if inst['answers'][-1][1] + 1 < len(inst['tokens']) and \
                                inst['text'][inst['tokens'][inst['answers'][-1][0]][1]: \
                                inst['tokens'][inst['answers'][-1][1] + 1][1]].strip() != answer['text']:
                            assert ValueError()
                inst['metadata'] = qa_metadata
                instances.append(inst)

                window_start_token_offset += STRIDE
        return instances


    def gen_question_instances(self, header, question_instances):


        if self._all_question_instances_in_batch:
            instances_to_add = question_instances
        else:
            # choose at most 2 instances from the same question:
            if len(question_instances) > 1 and self._use_one_inst_per_question:
                inst_with_answers = [inst for inst in question_instances if inst['answers'] != []]
                if len(inst_with_answers) > 0:
                    instances_to_add = random.sample(inst_with_answers, 1)
                else:
                    instances_to_add = []
            elif len(question_instances) > 2:
                # This part is inspired by Clark and Gardner, 17 - oversample the highest ranking documents.
                # In thier work they use only instances with answers, so we will find the highest
                # ranking instance with an answer (this also insures we have at least one answer in the chosen instances)

                inst_with_answers = [inst for inst in question_instances if inst['answers'] != []]
                instances_to_add = random.sample(inst_with_answers[0:2], 1)
                # we assume each question will be visited once in an epoch
                question_instances.remove(instances_to_add[0])
                instances_to_add += random.sample(question_instances, 1)

            else:
                instances_to_add = question_instances

            # Require at least one answer:
            #if not any(inst['answers'] != [] for inst in instances_to_add):
            #    raise ValueError()

        #logger.info("multiqa+: yielding %d instances ", len(filtered_instances))
        for inst_num, inst in enumerate(instances_to_add):
            tokenized_paragraph = [Token(text=t[0], idx=t[1]) for t in inst['tokens']]
            question_tokens = [Token(text=t[0], idx=t[1]) for t in inst['question_tokens']]
            new_passage = inst['text']
            new_answers = inst['answers']
            instance = util.make_reading_comprehension_instance_multiqa(question_tokens,
                                                                                 tokenized_paragraph,
                                                                                 self._token_indexers,
                                                                                 new_passage,
                                                                                 new_answers,
                                                                                 inst['metadata'],
                                                                                 header)

            yield instance

    @profile
    @overrides
    def _read(self, file_path: str):
        logger.info("Reading the dataset")

        # supporting multi dataset training:
        datasets = []
        for ind, single_file_path in enumerate(file_path.split(',')):
            single_file_path_cached = cached_path(single_file_path)
            zip_handle = zipfile.ZipFile(single_file_path_cached, 'r')
            datasets.append({'single_file_path':single_file_path, 'zip_handle':zip_handle, \
                             'file_handle': zip_handle.open(zip_handle.namelist()[0]), \
                             'num_of_questions':0, 'inst_remainder':[], \
                             'sample_ratio':1 if self._sampling_ratio is None else self._sampling_ratio[ind] })
            datasets[ind]['header'] = json.loads(datasets[ind]['file_handle'].readline())['header']

        is_done = [False for _ in datasets]
        while not all(is_done):
            for ind, dataset in enumerate(datasets):
                if is_done[ind]:
                    continue

                instances = []
                iter_question_count = 0
                for example in dataset['file_handle']:
                    unproc_example = json.loads(example)

                    instances += self.make_instances(unproc_example, datasets[ind]['header'])

                    # MRQA uses all questions:
                    dataset['header']['preproc.final_qas_used_fraction'] = 1.0

                    instance_list = sorted(instances, key=lambda x: x['metadata']['question_id'])
                    intances_question_id = [instance['metadata']['question_id'] for instance in instances]
                    split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
                    per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in range(len(split_inds) - 1)]

                    for question_instance in per_question_instances:
                        for instance in self.gen_question_instances(dataset['header'], question_instance):
                            yield instance
                    dataset['num_of_questions'] += len(per_question_instances)
                    iter_question_count += 1

                    if iter_question_count >= dataset['sample_ratio']:
                        break

                else:
                    # No more lines to be read from file
                    if self._rewind_datasets:
                        logger.info('rewinding! %s', dataset['single_file_path'])
                        # Reopening file (seek doesn't seem to work inside a zip)
                        dataset['file_handle'].close()
                        dataset['file_handle'] = dataset['zip_handle'].open(dataset['zip_handle'].namelist()[0])
                        # Reading header
                        _ = dataset['file_handle'].readline()
                    else:
                        is_done[ind] = True

                # per dataset sampling
                if self._sample_size > -1 and dataset['num_of_questions'] >= self._sample_size:
                    is_done[ind] = True

        for dataset in datasets:
            dataset['file_handle'].close()
            dataset['zip_handle'].close()



import json
import logging
from typing import Any, Dict, List, Tuple
import gzip,re, copy, random
import numpy as np

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, ListField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("mrqa_reader")
class MRQAReader(DatasetReader):
    """
    Reads MRQA formatted datasets files, and creates AllenNLP instances.
    This code supports comma separated list of datasets to perform Multi-Task training.
    Each instance is a single Question-Answer-Chunk. A Chunk is a single context for the model, where
    long contexts are split into multiple Chunks (usually of 512 word pieces for BERT), using sliding window.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sampling_ratio = None,
                 lazy: bool = False,
                 is_training = False,
                 sample_size: int = -1,
                 STRIDE: int = 128,
                 MAX_WORDPIECES: int = 512,
                 ) -> None:
        super().__init__(lazy)

        # make sure sampling can always be reproduced
        random.seed(0)

        self._STRIDE = STRIDE
        # NOTE AllenNLP automatically adds [CLS] and [SEP] word peices in the begining and end of the context,
        # therefore we need to subtract 2
        self._MAX_WORDPIECES = MAX_WORDPIECES - 2
        self._tokenizer = tokenizer or WordTokenizer()
        self._sampling_ratio = sampling_ratio
        self._sample_size = sample_size
        self._is_training = is_training
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

    def make_chunks(self, unproc_context, header, _bert_do_lowercase=True):
        """
        Each instance is a single Question-Answer-Chunk. A Chunk is a single context for the model, where
        long contexts are split into multiple Chunks (usually of 512 word pieces for BERT), using sliding window.

        :param unproc_context:
        :param header:
        :param _bert_do_lowercase:
        :return:
        """

        # converting the context into word pieces

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of 128.
        # splitting into chuncks that are not larger than 510 token pieces (NOTE AllenNLP
        # adds the [CLS] and [SEP] token pieces automatically
        per_question_chunks = []
        for qa in unproc_context['qas']:
            # is_impossible not supported at this point...
            if qa['is_impossible']:
                continue

            chunks = []
            curr_token_ix = 0
            window_start_token_offset = 0
            while curr_token_ix < len(unproc_context['context_tokens']):
                _, num_of_question_wordpieces = self.tokens_to_wordpieces(qa['question_tokens'])
                curr_token_ix = window_start_token_offset

                curr_context_tokens = qa['question_tokens'] + [['[SEP]',len(qa['question']) + 1]]
                context_char_offset = unproc_context['context_tokens'][curr_token_ix][1]
                question_char_offset = len(qa['question']) + 5 + 1 + 1
                num_of_wordpieces = 0
                while num_of_wordpieces < self._MAX_WORDPIECES - num_of_question_wordpieces - 1 \
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
                    if num_of_wordpieces < self._MAX_WORDPIECES - num_of_question_wordpieces - 1:
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
                chunks.append(inst)

                window_start_token_offset += self._STRIDE

            per_question_chunks.append(chunks)
        return per_question_chunks


    def gen_question_instances(self, header, question_instances):
        if self._is_training:
            # When training randomly choose one chunk per example (training with shared norm (Clark and Gardner, 17)
            # is not well defined when using sliding window )
            instances_to_add = random.sample(question_instances, 1)
        else:
            instances_to_add = question_instances

        for inst_num, inst in enumerate(instances_to_add):
            tokenized_paragraph = [Token(text=t[0], idx=t[1]) for t in inst['tokens']]
            question_tokens = [Token(text=t[0], idx=t[1]) for t in inst['question_tokens']]
            new_passage = inst['text']
            new_answers = inst['answers']
            instance = make_multiqa_instance(question_tokens,
                                             tokenized_paragraph,
                                             self._token_indexers,
                                             new_passage,
                                             new_answers,
                                             inst['metadata'],
                                             header)

            yield instance

    @overrides
    def _read(self, file_path: str):
        # supporting multi-dataset training:
        datasets = []
        for ind, single_file_path in enumerate(file_path.split(',')):
            single_file_path_cached = cached_path(single_file_path)
            zip_handle = gzip.open(single_file_path_cached, 'rb')
            #zip_handle = zipfile.ZipFile(single_file_path_cached, 'r')
            datasets.append({'single_file_path':single_file_path, 'zip_handle':zip_handle, \
                             'file_handle': zip_handle, \
                             'num_of_questions':0, 'inst_remainder':[], \
                             'num_to_sample':1 if self._sampling_ratio is None else self._sampling_ratio[ind] })
            datasets[ind]['header'] = json.loads(datasets[ind]['file_handle'].readline())['header']

        is_done = [False for _ in datasets]
        while not all(is_done):
            for ind, dataset in enumerate(datasets):
                if is_done[ind]:
                    continue

                for example in dataset['file_handle']:
                    for question_chunks in self.make_chunks(json.loads(example), datasets[ind]['header']):
                        for instance in self.gen_question_instances(dataset['header'], question_chunks):
                            yield instance
                        dataset['num_of_questions'] += 1

                    # supporting sampling of first #dataset['num_to_sample'] examples
                    if dataset['num_of_questions'] >= dataset['num_to_sample']:
                        break

                else:
                    # No more lines to be read from file
                    is_done[ind] = True

                # per dataset sampling
                if self._sample_size > -1 and dataset['num_of_questions'] >= self._sample_size:
                    is_done[ind] = True

        for dataset in datasets:
            logger.info("Total number of processed questions for %s is %d",dataset['header']['dataset'], dataset['num_of_questions'])
            dataset['file_handle'].close()
            dataset['zip_handle'].close()


def make_multiqa_instance(question_tokens: List[Token],
                                             tokenized_paragraph: List[List[Token]],
                                             token_indexers: Dict[str, TokenIndexer],
                                             paragraph: List[str],
                                             answers_list: List[Tuple[int, int]] = None,
                                             additional_metadata: Dict[str, Any] = None,
                                             header = None,
                                             use_multi_label_loss=False) -> Instance:
    """
    Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
    in a reading comprehension model.

    Note, this should be part of reading_comprehension/util.py

    Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
    ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
    and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
    fields, which are both ``IndexFields``.

    Parameters
    ----------
    question_list_tokens : ``List[List[Token]]``
        An already-tokenized list of questions. Each dialog have multiple questions.
    passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual span from the
        original passage that the model predicts as the answer to the question.  This is used in
        official evaluation scripts.
    token_spans_lists : ``List[List[Tuple[int, int]]]``, optional
        Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
        a list of list, first because there is multiple questions per dialog, and
        because there might be several possible correct answer spans in the passage.
        Currently, we just select the last span in this list (i.e., QuAC has multiple
        annotations on the dev set; this will select the last span, which was given by the original annotator).
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}

    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_paragraph]
    # This is separate so we can reference it later with a known type.
    passage_field = TextField(tokenized_paragraph, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    metadata = {'original_passage': paragraph,
                'answers_list': answers_list,
                'token_offsets': passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in tokenized_paragraph]}

    if answers_list is not None:
        if use_multi_label_loss:
            span_start_list: List[Field] = []
            span_end_list: List[Field] = []
            if answers_list == []:
                span_start_list.append(IndexField(-1, passage_field))
                span_end_list.append(IndexField(-1, passage_field))
            else:
                for answer in answers_list:
                    span_start_list.append(IndexField(answer[0], passage_field))
                    span_end_list.append(IndexField(answer[1], passage_field))


            fields['span_start'] = ListField(span_start_list)
            fields['span_end'] = ListField(span_end_list)

        else:
            span_start_list: List[Field] = []
            span_end_list: List[Field] = []
            if answers_list == []:
                span_start, span_end = -1, -1
            else:
                span_start, span_end, text = answers_list[0]

            span_start_list.append(IndexField(span_start, passage_field))
            span_end_list.append(IndexField(span_end, passage_field))

            fields['span_start'] = ListField(span_start_list)
            fields['span_end'] = ListField(span_end_list)

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)



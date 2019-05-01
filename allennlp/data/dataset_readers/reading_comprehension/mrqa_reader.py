import json
import logging
from typing import Any, Dict, List, Tuple
import gzip , copy, random

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
    Each instance is a single Question-Answer-Chunk. (This is will be changed in the future to instance per questions)
    A Chunk is a single context for the model, where long contexts are split into multiple Chunks (usually of 512 word pieces for BERT),
    using sliding window.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 dataset_weight = None,
                 lazy: bool = False,
                 is_training = False,
                 sample_size: int = -1,
                 STRIDE: int = 128,
                 MAX_WORDPIECES: int = 512,
                 ) -> None:
        super().__init__(lazy)

        # make sure results may be reproduced when sampling...
        random.seed(0)

        self._STRIDE = STRIDE
        # NOTE AllenNLP automatically adds [CLS] and [SEP] word peices in the begining and end of the context,
        # therefore we need to subtract 2
        self._MAX_WORDPIECES = MAX_WORDPIECES - 2
        self._tokenizer = tokenizer or WordTokenizer()
        self._dataset_weight = dataset_weight
        self._sample_size = sample_size
        self._is_training = is_training

        # BERT specific init
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self._bert_wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer.tokenize
        self._never_lowercase = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

    def token_to_wordpieces(self, token, _bert_do_lowercase=True):
        text = (token[0].lower()
                if _bert_do_lowercase and token[0] not in self._never_lowercase
                else token[0])
        word_pieces = self._bert_wordpiece_tokenizer(text)
        return len(word_pieces), word_pieces

    def make_chunks(self, unproc_context, header, _bert_do_lowercase=True):
        """
        Each instance is a single Question-Answer-Chunk. A Chunk is a single context for the model, where
        long contexts are split into multiple Chunks (usually of 512 word pieces for BERT), using sliding window.
        """

        # We could have contexts that are longer than the maximum sequence length.
        # To tackle this we'll implement a sliding window approach, where we take chunks
        # of up to our max length with a fixed stride.
        # splitting into chuncks that are not larger than 510 token pieces (NOTE AllenNLP
        # adds the [CLS] and [SEP] token pieces automatically)
        per_question_chunks = []
        for qa in unproc_context['qas']:
            chunks = []
            curr_token_ix = 0
            window_start_token_offset = 0
            while curr_token_ix < len(unproc_context['context_tokens']):
                num_of_question_wordpieces = 0
                for token in qa['question_tokens']:
                    num_of_wordpieces, _ = self.token_to_wordpieces(token)
                    num_of_question_wordpieces += num_of_wordpieces

                curr_token_ix = window_start_token_offset
                if curr_token_ix >= len(unproc_context['context_tokens']):
                    continue

                curr_context_tokens = qa['question_tokens'] + [['[SEP]',len(qa['question']) + 1]]
                context_char_offset = unproc_context['context_tokens'][curr_token_ix][1]
                # 5 chars for [SEP], 1 + 1 chars for spaces
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

                    _, word_pieces = self.token_to_wordpieces(curr_token)

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
                    if answer['token_spans'][0][0] >= window_start_token_offset and \
                        answer['token_spans'][0][1] < window_end_token_offset:
                        qa_metadata['has_answer'] = True
                        answer_token_offset = len(qa['question_tokens']) + 1 - window_start_token_offset
                        inst['answers'].append((answer['token_spans'][0][0] + answer_token_offset, \
                                                       answer['token_spans'][0][1] + answer_token_offset,
                                                       answer['text']))

                inst['metadata'] = qa_metadata
                chunks.append(inst)

                window_start_token_offset += self._STRIDE

            if len([inst for inst in chunks if inst['answers'] != []])>0:
                per_question_chunks.append(chunks)
        return per_question_chunks

    def gen_question_instances(self, question_chunks):
        if self._is_training:
            # When training randomly choose one chunk per example (training with shared norm (Clark and Gardner, 17)
            # is not well defined when using sliding window )
            chunks_with_answers = [inst for inst in question_chunks if inst['answers'] != []]
            instances_to_add = random.sample(chunks_with_answers, 1)
        else:
            instances_to_add = question_chunks

        for inst_num, inst in enumerate(instances_to_add):
            instance = make_multiqa_instance([Token(text=t[0], idx=t[1]) for t in inst['question_tokens']],
                                             [Token(text=t[0], idx=t[1]) for t in inst['tokens']],
                                             self._token_indexers,
                                             inst['text'],
                                             inst['answers'],
                                             inst['metadata'])
            yield instance

    @overrides
    def _read(self, file_path: str):
        # supporting multi-dataset training:
        datasets = []
        for ind, single_file_path in enumerate(file_path.split(',')):
            single_file_path_cached = cached_path(single_file_path)
            zip_handle = gzip.open(single_file_path_cached, 'rb')
            datasets.append({'single_file_path':single_file_path, \
                             'file_handle': zip_handle, \
                             'num_of_questions':0, 'inst_remainder':[], \
                             'dataset_weight':1 if self._dataset_weight is None else self._dataset_weight[ind] })
            datasets[ind]['header'] = json.loads(datasets[ind]['file_handle'].readline())['header']

        is_done = [False for _ in datasets]
        while not all(is_done):
            for ind, dataset in enumerate(datasets):
                if is_done[ind]:
                    continue

                for example in dataset['file_handle']:
                    for question_chunks in self.make_chunks(json.loads(example), datasets[ind]['header']):
                        for instance in self.gen_question_instances( question_chunks):
                            yield instance
                        dataset['num_of_questions'] += 1

                    # supporting sampling of first #dataset['num_to_sample'] examples
                    if dataset['num_of_questions'] >= dataset['dataset_weight']:
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


def make_multiqa_instance(question_tokens: List[Token],
                                             tokenized_paragraph: List[List[Token]],
                                             token_indexers: Dict[str, TokenIndexer],
                                             paragraph: List[str],
                                             answers_list: List[Tuple[int, int]] = None,
                                             additional_metadata: Dict[str, Any] = None) -> Instance:

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



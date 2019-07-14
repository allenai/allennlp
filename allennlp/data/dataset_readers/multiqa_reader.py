import logging
from typing import Any, Dict, List, Tuple
import copy, random
import uuid
import os
import sys
import json
import tqdm
import pathlib
import spacy
import boto3
import gzip
import hashlib
import string
import numpy as np
from multiprocessing import Pool, cpu_count

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, ListField, LabelField
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords

# ALON - for line profiler
try:
    profile
except NameError:
    profile = lambda x: x

# Globals
SEPARATORS = ['[PAR]', '[DOC]', '[TLE]']

# Process-local tokenizer
spacy_tok = None

# Punctuation to be stripped
STRIP_CHARS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’´`_'

class NltkPlusStopWords():
    """ Configurablable access to stop word """

    def __init__(self, punctuation=False):
        self._words = None
        self.punctuation = punctuation

    @property
    def words(self):
        if self._words is None:
            self._words = set(stopwords.words('english'))
            # Common question words we probably want to ignore, "de" was suprisingly common
            # due to its appearance in person names
            self._words.update(["many", "how", "de"])
            if self.punctuation:
                self._words.update(string.punctuation)
                self._words.update(["£", "€", "¥", "¢", "₹", "\u2212",
                                    "\u2014", "\u2013", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0"])
        return self._words

class Paragraph_TfIdf_Scoring():
    # Hard coded weight learned from a logistic regression classifier
    TFIDF_W = 5.13365065
    LOG_WORD_START_W = 0.46022765
    FIRST_W = -0.08611607
    LOWER_WORD_W = 0.0499123
    WORD_W = -0.15537181

    def __init__(self):
        self._stop = NltkPlusStopWords(True).words
        self._stop.remove('퀜')
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self._stop)

    def score_paragraphs(self, question, paragraphs):
        tfidf = self._tfidf
        text = paragraphs
        para_features = tfidf.fit_transform(text)
        q_features = tfidf.transform(question)

        q_words = {x for x in question if x.lower() not in self._stop}
        q_words_lower = {x.lower() for x in q_words}
        word_matches_features = np.zeros((len(paragraphs), 2))
        for para_ix, para in enumerate(paragraphs):
            found = set()
            found_lower = set()
            for sent in para:
                for word in sent:
                    if word in q_words:
                        found.add(word)
                    elif word.lower() in q_words_lower:
                        found_lower.add(word.lower())
            word_matches_features[para_ix, 0] = len(found)
            word_matches_features[para_ix, 1] = len(found_lower)

        tfidf = pairwise_distances(q_features, para_features, "cosine").ravel()
        # TODO 0 represents if this paragraph start a real paragraph (number > 0 represents the
        # paragraph was split. when we split paragraphs we need to take care of this...
        starts = np.array([0 for p in paragraphs])
        log_word_start = np.log(starts/400.0 + 1)
        first = starts == 0
        scores = tfidf * self.TFIDF_W + self.LOG_WORD_START_W * log_word_start + self.FIRST_W * first +\
                 self.LOWER_WORD_W * word_matches_features[:, 1] + self.WORD_W * word_matches_features[:, 0]
        return scores

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("multiqa_reader")
class MultiQAReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 dataset_weight = None,
                 preproc_outputfile: str = None,
                 n_processes: int = None,
                 lazy: bool = False,
                 is_training = False,
                 sample_size: int = -1,
                 STRIDE: int = 128,
                 MAX_WORDPIECES: int = 512,
                 random_seed: int = 0,
                 support_yesno: bool = True
                 ) -> None:
        super().__init__(lazy)

        # the random seed could be change for models like BERT that are
        # unstable when fine-tuned + the insure results reproducibility
        random.seed(random_seed)

        self._support_yesno = support_yesno
        self._preproc_outputfile = preproc_outputfile
        self._STRIDE = STRIDE
        # NOTE AllenNLP automatically adds [CLS] and [SEP] word peices in the begining and end of the context,
        # therefore we need to subtract 2
        self._MAX_WORDPIECES = MAX_WORDPIECES - 2
        self._tokenizer = tokenizer or WordTokenizer()
        self._dataset_weight = dataset_weight
        self._sample_size = sample_size
        self._is_training = is_training
        self._n_processes = n_processes

        # BERT specific init
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self._bert_wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer.tokenize
        self._never_lowercase = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

        self.sep_tokens = {'title':'[TLE]', 'text':'[DOC]'}
        self._para_tfidf_scoring = Paragraph_TfIdf_Scoring

    def token_to_wordpieces(self, token, _bert_do_lowercase=True):
        text = (token[0].lower()
                if _bert_do_lowercase and token[0] not in self._never_lowercase
                else token[0])
        word_pieces = self._bert_wordpiece_tokenizer(text)
        return len(word_pieces), word_pieces

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
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
        tok_answer_text = " ".join(tokenizer(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    @profile
    def combine_context(self, context):
        ## reording the documents using TF-IDF distance to the questions
        # documents are usually from different sources so there order can usually be modified
        #docs = [doc['title'] + ' ' + doc['text'] for doc in context['context']['documents']]
        #doc_scores = self._para_tfidf_scoring.score_paragraphs([context['qas'][0]['question']], docs)

        offsets = []
        context_tokens = []
        context_text = ''
        text_offset = 0
        token_offset = 0
        for doc_ind, document in enumerate(context['context']['documents']):
            offsets.append({})
            for part in ['title','text']:
                if part in document:
                    # adding the separator
                    context_text += self.sep_tokens[part] + ' '
                    context_tokens.append((self.sep_tokens[part], text_offset))
                    text_offset += len(self.sep_tokens[part]) + 1
                    token_offset += 1

                    # updating offsets for changing the answer offset byte and token num
                    offsets[doc_ind][part] = {}
                    offsets[doc_ind][part]['token_offset'] = token_offset
                    offsets[doc_ind][part]['text_offset'] = text_offset

                    # adding the actual text and tokens
                    context_text += document[part] + ' '
                    context_tokens += [(t[0],t[1] + text_offset) for t in document['tokens'][part]]
                    text_offset += len(document[part]) + 1
                    token_offset += len(document['tokens'][part])

        context['full_text'] = context_text
        context['context_tokens'] = context_tokens

        # updating answer tokens offsets and text
        no_answer_questions = []
        for qa in context['qas']:
            answer_text_list = []
            qa['detected_answers'] = []
            qa['yesno'] = 'no_yesno'
            qa['cannot_answer'] = False
            if 'open-ended' in qa['answers']:
                if 'answer_candidates' in qa['answers']['open-ended']:
                    for ac in qa['answers']['open-ended']["answer_candidates"]:
                        if 'extractive' in ac:
                            # Supporting only one answer of type extractive (future version will support list and set)
                            if "single_answer" in ac['extractive']:
                                for instance in ac['extractive']["single_answer"]["instances"]:
                                    detected_answer = {}
                                    # checking if the answer has been detected
                                    if "token_offset" in offsets[instance["doc_id"]][instance["part"]]:
                                        answer_token_offset = offsets[instance["doc_id"]][instance["part"]]['token_offset']
                                        detected_answer["token_spans"] = (instance['token_inds'][0] + answer_token_offset,
                                                                          instance['token_inds'][1] + answer_token_offset)
                                        detected_answer['text'] = instance["text"]
                                        qa['detected_answers'].append(detected_answer)
                                    answer_text_list.append(instance["text"])
                        elif 'yesno' in ac:
                            # Supporting only one answer of type yesno
                            if self._support_yesno and "single_answer" in ac['yesno']:
                                qa['yesno'] = ac['yesno']['single_answer']

                elif 'cannot_answer' in qa['answers']['open-ended']:
                    qa['cannot_answer'] = True

            qa['answer_text_list'] = answer_text_list

        return context

    @profile
    def make_chunks(self, unproc_context, header, _bert_do_lowercase=True):
        """
        Each instance is a single Question-Answer-Chunk. A Chunk is a single context for the model, where
        long contexts are split into multiple Chunks (usually of 512 word pieces for BERT), using sliding window.
        """

        # unify the whole context (and save the document start bytes and tokens)
        # for each now create a question + context for each train set with answer
        # and question + context for each dev set...

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
                    curr_token = unproc_context['context_tokens'][curr_token_ix]

                    # BERT has only [SEP] in it's word piece vocabulary. because we keps all separators char length 5
                    # we can replace all of them with [SEP] without modifying the offset
                    curr_token_text = curr_token[0]
                    if curr_token_text in ['[TLE]','[PAR]','[DOC]']:
                        curr_token_text = '[SEP]'

                    # fixing the car offset of each token
                    curr_token_offset = curr_token[1]
                    curr_token_offset += question_char_offset - context_char_offset

                    _, word_pieces = self.token_to_wordpieces(curr_token)

                    num_of_wordpieces += len(word_pieces)
                    if num_of_wordpieces < self._MAX_WORDPIECES - num_of_question_wordpieces - 1:
                        window_end_token_offset = curr_token_ix + 1
                        curr_context_tokens.append((curr_token_text,curr_token_offset))
                    curr_token_ix += 1


                inst = {}
                inst['yesno'] = qa['yesno']
                inst['cannot_answer'] = qa['cannot_answer']
                inst['question_tokens'] = qa['question_tokens']
                inst['tokens'] = curr_context_tokens
                inst['text'] = qa['question'] + ' [SEP] ' + unproc_context['full_text'][context_char_offset: \
                            context_char_offset + curr_context_tokens[-1][1] + len(curr_context_tokens[-1][0]) + 1]
                inst['answers'] = []
                qa_metadata = {'has_answer': False, 'dataset': header['dataset_name'], "question_id": qa['qid'], \
                               'answer_texts_list': list(set(qa['answer_text_list']))}
                for answer in qa['detected_answers']:
                    if len(answer['token_spans']) > 0 and answer['token_spans'][0] >= window_start_token_offset and \
                        answer['token_spans'][1] < window_end_token_offset:
                        qa_metadata['has_answer'] = True
                        answer_token_offset = len(qa['question_tokens']) + 1 - window_start_token_offset

                        (tok_start_position, tok_end_position) = self._improve_answer_span([t[0] for t in inst['tokens']], \
                                answer['token_spans'][0] + answer_token_offset, \
                                answer['token_spans'][1] + answer_token_offset, self._bert_wordpiece_tokenizer, answer['text'])

                        inst['answers'].append((tok_start_position, tok_end_position, answer['text']))

                inst['metadata'] = qa_metadata
                chunks.append(inst)

                window_start_token_offset += self._STRIDE

            per_question_chunks.append(chunks)

        return per_question_chunks

    @profile
    def gen_question_instances(self, question_chunks):
        instances_to_add = []
        if self._is_training:
            # Trying to balance the chunks with answer and the ones without by sampling one from each
            # if each is available
            cannot_answer = [inst for inst in question_chunks if inst['cannot_answer']]
            yesno = [inst for inst in question_chunks if inst['yesno'] != 'no_yesno']
            spans = [inst for inst in question_chunks if len(inst['answers']) > 0]
            chunks_to_select_from = cannot_answer + yesno + spans
            if len(chunks_to_select_from) > 0:
                instances_to_add += random.sample(chunks_to_select_from, 1)

        else:
            instances_to_add = question_chunks

        for inst_num, inst in enumerate(instances_to_add):
            instance = make_multiqa_instance([Token(text=t[0], idx=t[1]) for t in inst['question_tokens']],
                                             [Token(text=t[0], idx=t[1]) for t in inst['tokens']],
                                             self._token_indexers,
                                             inst['text'],
                                             inst['answers'],
                                             inst['yesno'],
                                             inst['metadata'])
            yield instance

    @profile
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
                    example = self.combine_context(json.loads(example))

                    for question_chunks in self.make_chunks(example, datasets[ind]['header']):
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
            logger.info("Total number of processed questions for %s is %d",dataset['header']['dataset_name'], dataset['num_of_questions'])
            dataset['file_handle'].close()


def make_multiqa_instance(question_tokens: List[Token],
                             tokenized_paragraph: List[List[Token]],
                             token_indexers: Dict[str, TokenIndexer],
                             paragraph: List[str],
                             answers_list: List[Tuple[int, int]] = None,
                             yesno: List[str] = None,
                             additional_metadata: Dict[str, Any] = None) -> Instance:

    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}

    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_paragraph]
    # This is separate so we can reference it later with a known type.
    passage_field = TextField(tokenized_paragraph, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    fields['yesno_labels'] = LabelField(yesno, label_namespace="yesno_labels")
    metadata = {'original_passage': paragraph,
                'answers_list': answers_list,
                'cannot_answer': False,
                'token_offsets': passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in tokenized_paragraph]}

    if answers_list is not None:
        span_start_list: List[Field] = []
        span_end_list: List[Field] = []
        if answers_list == []:
            # No answer will point at the beginning of the chunk (see
            # A BERT Baseline for the Natural Questions https://arxiv.org/abs/1901.08634 )
            # Note connot_answer (SQuAD 2.0) and negative examples (just example in which we could not find
            #the gold answer, if any are used), are both treated similarly here.
            span_start, span_end = 0, 0
            metadata['cannot_answer'] = True
        else:
            span_start, span_end, text = answers_list[0]

        span_start_list.append(IndexField(span_start, passage_field))
        span_end_list.append(IndexField(span_end, passage_field))

        fields['span_starts'] = ListField(span_start_list)
        fields['span_ends'] = ListField(span_end_list)

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)



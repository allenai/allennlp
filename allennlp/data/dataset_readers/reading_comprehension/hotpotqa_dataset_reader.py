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
    MetadataField, ListField

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

def warn(message):
    print('WARNING: %s' % message, file=sys.stderr)

def read_corpus(split):
    if split == 'train':
        single_file_path = cached_path("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json")
    elif split == 'dev':
        single_file_path = cached_path("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")

    with open(single_file_path, 'r') as myfile:
        original_dataset = json.load(myfile)

    data = original_dataset
    contexts = []
    for example in tqdm.tqdm(data, total=len(data), ncols=80):
        # choosing only the gold paragraphs
        gold_paragraphs = []
        for supp_fact in example['supporting_facts']:
            for context in example['context']:
                # finding the gold context
                if context[0] == supp_fact[0]:
                    gold_paragraphs.append(context)

        context = ''
        for para in gold_paragraphs:
            context += '[PAR] [TLE] ' + para[0] + ' [SEP] '
            context += ' '.join(para[1]) + ' '
        answers = [{'text': example['answer']}]

        qas = [{"id": example['_id'] + "#0",
                "question": example['question'],
                "answers": answers,
                }]

        contexts.append({"id": example['_id'], "context": context, "qas": qas})
    return contexts

def bool_flag(string):
    string = string.lower()
    if string in ['1', 'true', 'yes', 't', 'y']:
        return True
    if string in ['0', 'false', 'no', 'f', 'n']:
        return False
    raise ValueError('Unknown boolean option %s' % string)

def get_hash(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def update_manifest(name, stats):
    """Write statistics to manifest.json"""
    root = os.path.dirname(os.path.dirname(__file__))
    fname = os.path.join(root, 'manifest.json')
    if os.path.exists(fname):
        with open(fname) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest[name] = stats
    with open(fname, 'w') as f:
        json.dump(manifest, f, sort_keys=True, indent=4)
    s3 = boto3.client('s3')
    s3.upload_file(fname, 'mrqa', 'data/manifest.json', ExtraArgs={'ACL': 'public-read'})

def get_tokenizer():
    """Load spacy tokenizer with special tokenization cases."""
    en = spacy.load('en', disable=['tagger', 'parser', 'entity'])
    tok = en.tokenizer
    for sep in SEPARATORS:
        tok.add_special_case(sep, [{spacy.attrs.ORTH: sep}])
    return tok

def init_worker():
    """Init function to be run at start-up during multiprocessing."""
    # Tokenizer instances will be local to the process.
    global spacy_tok
    spacy_tok = get_tokenizer()

def tokenize(string):
    """Compute both tokens and word pieces."""
    global spacy_tok
    tokens = spacy_tok(string)
    return tokens

def filter(question, answer):
    """Takes in tokenized question and answer and decides whether to skip."""
    if len(answer) == 1:
        if answer[0].text.lower() in ['true', 'false', 'yes', 'no']:
            return True
    return False

def find_all_spans(context, answer):
    """Find all exact matches of `answer` in `context`.
    - Matches are case, article, and punctuation insensitive.
    - Matching follows SQuAD eval protocol.
    - The context and answer are assumed to be tokenized
      (using either tokens or word pieces).
    Returns [start, end] (inclusive) token span.
    """
    # Lower-case and strip all tokens.
    words = [t.lower().strip(STRIP_CHARS) for t in context]
    answer = [t.lower().strip(STRIP_CHARS) for t in answer]

    # Strip answer empty tokens + articles
    answer = [t for t in answer if t not in {'', 'a', 'an', 'the'}]
    if len(answer) == 0:
        return []

    # Find all possible starts (matches first answer token).
    occurences = []
    word_starts = [i for i, w in enumerate(words) if answer[0] == w]
    n_tokens = len(answer)

    # Advance forward until we find all the words, skipping over articles
    for start in word_starts:
        end = start + 1
        ans_token = 1
        while ans_token < n_tokens and end < len(words):
            next = words[end]
            if answer[ans_token] == next:
                ans_token += 1
                end += 1
            elif next in {'', 'a', 'an', 'the'}:
                end += 1
            else:
                break
        if n_tokens == ans_token:
            occurences.append((start, end - 1))
    return list(set(occurences))

def char_to_token(tokens, char_start, char_end):
    """Map character offsets to token offsets."""
    start = None
    end = None
    for i, t in enumerate(tokens):
        if t.idx == char_start:
            start = i
        if t.idx + len(t.text) - 1 == char_end:
            end = i

    if start is None or end is None:
        return None

    return (start, end)

def process_example(example):
    """Process a single context with question example.
    1) Find answer spans for question pairs (that don't already have).
    2) Discard questions that don't have exact match spans.
    """
    # Record stats on filtered questions.
    num_skipped = 0

    # Skip zero length contexts
    #if len(example['context']) == 0:
    #    return None, len(example['qas'])

    # Truncate
    context_tokens = tokenize(example['context'])
    context = ''.join([t.text_with_ws for t in context_tokens])
    example['context'] = context

    # Filter whitespace tokens: https://github.com/explosion/spaCy/issues/1707
    context_tokens = [t for t in context_tokens if not t.is_space]
    example['context_tokens'] = [(t.text, t.idx) for t in context_tokens]

    # Keep filtered list of question-answer pairs.
    kept_qas = []

    # Iterate questions, keeping ones with valid answers.
    # Tokenize questions as well.
    for qa in example['qas']:
        question_tokens = tokenize(qa['question'])
        qa['question_tokens'] = [(t.text, t.idx) for t in question_tokens]

        # If the question is unanswerable, record this and move on.
        if 'is_impossible' in qa and qa['is_impossible']:
            kept_qas.append(qa)

        # Otherwise get valid answer.
        else:
            qa['is_impossible'] = False
            detected_ans = []
            for answer in qa['answers']:
                # If span is provided, confirm it and move on.
                if 'answer_start' in answer and answer['answer_start']:
                    char_start = answer['answer_start']
                    char_end = answer['answer_start'] + len(answer['text']) - 1
                    answer['char_spans'] = [(char_start, char_end)]

                    # Enforce that it has to be a valid *tokenized* span.
                    token_span = char_to_token(context_tokens, char_start, char_end)
                    if not token_span:
                        continue

                    answer['token_spans'] = [token_span]
                    detected_ans.append(answer)

                # Otherwise, try to find it...
                else:
                    # Tokenize answer
                    text = answer['text']
                    answer_tokens = tokenize(text)

                    # Find spans (by token matches).
                    token_spans = find_all_spans([t.text for t in context_tokens],
                                                 [t.text for t in answer_tokens])

                    # Skip if could not find span
                    #if not token_spans:
                    #    continue
                    answer['token_spans'] = token_spans

                    char_spans = []
                    for start, end in token_spans:
                        char_start = context_tokens[start].idx
                        char_end = (context_tokens[end].idx +
                                    len(context_tokens[end].text) - 1)
                        char_spans.append((char_start, char_end))
                    answer['char_spans'] = char_spans

                    detected_ans.append(answer)

            # If there are any valid answers, keep the question.
            # Also keep all (potentially invalid / not exact span) answers.
            if len(detected_ans) > 0:
                qa['detected_answers'] = detected_ans
                qa['answers'] = [a['text'] for a in qa['answers']]
                kept_qas.append(qa)
            else:
                num_skipped += 1

    # If no valid qas, just return None.
    if len(kept_qas) == 0:
        return None, num_skipped

    example['qas'] = kept_qas

    return example, num_skipped

def process_and_dump(examples, filename , header=None, processes=None, upload=None):
    """Run over all examples, with multiprocessing.
    Examples should be provided in the form of an iterable.
    """
    if filename.startswith('s3'):
        dirname = '.'
        local_filename = 'temp.jsonl.gz'
        cloudbucket = filename.split('/')[2]
        cloudfile = '/'.join(filename.split('/')[3:])
    else:
        local_filename = filename
        dirname = '/'.join(filename.split('/')[0:-1])
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)

    if processes is None or processes > 1:
        process = processes or cpu_count() - 1
        workers = Pool(processes, initializer=init_worker)
        map_fn = workers.imap_unordered
    else:
        init_worker()
        map_fn = map
    try:
        N = len(examples)
    except:
        N = None

    stats = {'total': 0, 'original_skipped': 0, 'avg_context_tokens': 0, 'impossible': 0}
    with tqdm.tqdm(total=N) as pbar, gzip.open(local_filename, 'wb') as f:
        f.write((json.dumps({'header': header}) + '\n').encode('utf-8'))
        for ex, num_skipped in map_fn(process_example, examples):
            stats['original_skipped'] += num_skipped
            if ex is not None:
                stats['total'] += len(ex['qas'])
                stats['avg_context_tokens'] += len(ex['context_tokens']) * len(ex['qas'])
                output = json.dumps(ex) + '\n'
                f.write(output.encode('utf-8'))
            pbar.update()

    # Print stats
    stats['avg_context_tokens'] = round(stats['avg_context_tokens'] / stats['total'])
    original_total = stats['total'] + stats['original_skipped']
    print('Kept %d out of %d questions (%2.2f%%). Skipped %d.' %
          (stats['total'], original_total, stats['total'] / original_total * 100, stats['original_skipped']))

    # Upload to S3
    if filename.startswith('s3'):
        print('Uploading to S3, this may take some time ... ')
        s3 = boto3.client('s3')
        s3.upload_file(local_filename, cloudbucket, cloudfile, ExtraArgs={'ACL':'public-read'})
        os.remove(local_filename)

    ## TODO this is a hack, waiting for caching feature to be integrated in AllenNLP
    return Instance({})

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
                 preproc_outputfile: str = None,
                 n_processes: int = None,
                 lazy: bool = False,
                 is_training = False,
                 sample_size: int = -1,
                 STRIDE: int = 128,
                 MAX_WORDPIECES: int = 512,
                 ) -> None:
        super().__init__(lazy)

        # make sure results may be reproduced when sampling...
        random.seed(0)
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

    def token_to_wordpieces(self, token, _bert_do_lowercase=True):
        text = (token[0].lower()
                if _bert_do_lowercase and token[0] not in self._never_lowercase
                else token[0])
        word_pieces = self._bert_wordpiece_tokenizer(text)
        return len(word_pieces), word_pieces

    @profile
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
                inst['question_tokens'] = qa['question_tokens']
                inst['tokens'] = curr_context_tokens
                inst['text'] = qa['question'] + ' [SEP] ' + unproc_context['context'][context_char_offset: \
                            context_char_offset + curr_context_tokens[-1][1] + len(curr_context_tokens[-1][0]) + 1]
                inst['answers'] = []
                qa_metadata = {'has_answer': False, 'dataset': header['dataset'], "question_id": qa['id'], \
                               'answer_texts_list': list(set(qa['answers']))}
                for answer in qa['detected_answers']:
                    if len(answer['token_spans']) > 0 and answer['token_spans'][0][0] >= window_start_token_offset and \
                        answer['token_spans'][0][1] < window_end_token_offset:
                        qa_metadata['has_answer'] = True
                        answer_token_offset = len(qa['question_tokens']) + 1 - window_start_token_offset
                        inst['answers'].append((answer['token_spans'][0][0] + answer_token_offset, \
                                                       answer['token_spans'][0][1] + answer_token_offset,
                                                       answer['text']))

                inst['metadata'] = qa_metadata
                chunks.append(inst)

                window_start_token_offset += self._STRIDE

            # In training we need examples with answer only
            if not self._is_training or len([inst for inst in chunks if inst['answers'] != []])>0:
                per_question_chunks.append(chunks)
        return per_question_chunks

    @profile
    def gen_question_instances(self, question_chunks):
        instances_to_add = []
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
        if file_path.endswith('.json'):
            yield from self._read_examples_file(file_path)
        elif file_path.endswith('.jsonl.gz'):
            yield from self._read_preprocessed_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        split = 'train' if self._is_training else 'evaluation'
        header = {"dataset": 'HotpotQA'}

        single_file_path = cached_path(file_path)
        with open(single_file_path, 'r') as myfile:
            original_dataset = json.load(myfile)

        data = original_dataset
        contexts = []
        for example in tqdm.tqdm(data, total=len(data), ncols=80):
            # choosing only the gold paragraphs
            gold_paragraphs = []
            for supp_fact in example['supporting_facts']:
                for context in example['context']:
                    # finding the gold context
                    if context[0] == supp_fact[0]:
                        gold_paragraphs.append(context)

            context = ''
            for para in gold_paragraphs:
                context += '[PAR] [TLE] ' + para[0] + ' [SEP] '
                context += ' '.join(para[1]) + ' '
            answers = [{'text': example['answer']}]

            qas = [{"id": example['_id'] + "#0",
                    "question": example['question'],
                    "answers": answers,
                    }]

            if self._sample_size != -1 and len(contexts) > self._sample_size:
                break

            contexts.append({"id": example['_id'], "context": context, "qas": qas})

        yield process_and_dump(contexts, self._preproc_outputfile, header, processes=self._n_processes)

    @profile
    def _read_preprocessed_file(self, file_path: str):
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



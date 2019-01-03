import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random, math
import sys, os
import boto3
from typing import TypeVar,Iterable
from multiprocessing import Pool
        

T = TypeVar('T')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.common.util import add_noise_to_dict_values

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
import string

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

    def __init__(self, n_to_select):
        self.n_to_select = n_to_select
        self._stop = NltkPlusStopWords(True).words
        self._stop.remove('퀜')
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self._stop)

    def score_paragraphs(self, question, paragraphs):
        tfidf = self._tfidf
        text = paragraphs
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

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

    def prune(self, question, paragraphs):
        scores = self.score_paragraphs(question, paragraphs)
        sorted_ix = np.argsort(scores)

        return [paragraphs[i] for i in sorted_ix[:self.n_to_select]]

class MultiQAPreprocess():

    def __init__(self,
                 max_context_docs,
                 max_doc_size,
                 use_document_titles,
                 use_rank,
                 require_answer_in_doc,
                 require_answer_in_question) -> None:
        self._tokenizer = WordTokenizer()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._max_num_docs = max_context_docs
        self._max_doc_size = max_doc_size
        self._use_document_titles = use_document_titles
        self._require_answer_in_doc = require_answer_in_doc
        self._require_answer_in_question = require_answer_in_question
        self._para_tfidf_scoring = Paragraph_TfIdf_Scoring(max_context_docs)
        self._use_rank = use_rank
        self._answers_removed = 0
        self._total_answers = 0

        # we chose "^..^" because the tokenizer splits the standard "<..>" chars
        self._SEP = ' ^SEP^ '
        self._PARA_SEP = ' ^PARA^ '
        self._KNOWN_SEP = {'rank':' ', 'title':' ^TITLE_SEP^ '}
        
    def iterate_doc_parts(self,document):
        part_num = 0

        if self._use_rank:
            yield (part_num,'rank','^RANK^' + str(document['rank']))
            part_num += 1

        if self._use_document_titles:
            yield (part_num, 'title',document['title'])
            part_num += 1

        for ind,snippet in enumerate(document['snippets']):
            # using the special character "«" to indicate newline (new paragraph)
            # so that the spacy tokenizer will split paragraphs with a special token of length 1. 
            # (this preserves the all the answer_starts within a snippet...)
            snippet = snippet.replace("\n","«")
            yield (part_num, ind, snippet)
            part_num += 1

    def compute_token_answer_starts(self, qas, doc_ind, part_type,part_num, part_text, part_tokens):
        part_offsets = [(token[1], token[1] + len(token[0])) for token in part_tokens]
        for qa in qas:
            for answer in qa['answers']:
                for alias in answer['aliases']:
                    for alias_start in alias['answer_starts']:
                        if alias_start[0] == doc_ind and alias_start[1] == part_type:
                            char_span_start = alias_start[2]
                            char_span_end = char_span_start + len(alias['text'])
                            if 'token_answer_starts' not in alias:
                                    alias['token_answer_starts'] = []
                            try:
                                (span_start, span_end), error = util.char_span_to_token_span(part_offsets,
                                                                            (char_span_start, char_span_end))
                                alias['token_answer_starts'].append((doc_ind, part_num,span_start,span_end))
                            except:
                                self._answers_removed += 1
                                self._total_answers += 1
        return qas
    
    def update_answer_docid(self,new_qas, qas, curr_doc_ind, org_doc_ind ,part_num, org_part_num):
        for qa_ind, qa in enumerate(qas):
            for ans_ind, answer in enumerate(qa['answers']):
                for alias_ind, alias in enumerate(answer['aliases']):
                    for ind, alias_start in enumerate(alias['token_answer_starts']):
                        if alias_start[0] == org_doc_ind and alias_start[1] == org_part_num:
                           new_qas[qa_ind]['answers'][ans_ind]['aliases'][alias_ind]['token_answer_starts'][ind] \
                                = (curr_doc_ind, part_num, alias_start[2], alias_start[3])
        return new_qas
    
    def split_documents(self, context):
        new_qas = copy.deepcopy(context['qas'])
        new_documents = copy.deepcopy(context['documents'])
        for org_doc_ind, document in enumerate(context['documents']):
            if document['num_of_tokens'] > self._max_doc_size:
                token_cumsum = 0
                new_document = None
                for part_ind, part in enumerate(document['parts']):
                    # check if to split
                    if token_cumsum + len(part['tokens']) > self._max_doc_size:
                        #if snippet is larger than max_doc_size then split by "«" (we need a special function 
                        #for this so it can wait for tomorrow)  
                        if len(part['tokens']) > self._max_doc_size:
                            pass
                        
                        # splitting the original document
                        if not new_document:
                            new_documents[org_doc_ind]['num_of_tokens'] = token_cumsum
                            new_documents[org_doc_ind]['parts'] = document['parts'][0:part_ind] 

                        new_document = {'parts':[],'num_of_tokens':0,'org_doc':org_doc_ind}
                        new_documents.append(new_document)
                        token_cumsum = 0
                        curr_doc_ind = len(new_documents)-1
                    
                    if new_document:
                        new_qas = self.update_answer_docid(new_qas, context['qas'], curr_doc_ind, org_doc_ind, \
                            len(new_document['parts']), part_ind)
                        new_document['parts'].append(part)
                        new_document['num_of_tokens'] += len(part['tokens'])
                        

                    token_cumsum += len(part['tokens'])
        context['documents'] = new_documents
        context['qas'] = new_qas

    def tokenize_context(self, context):
        paragraphs = ['']
        curr_paragraph = 0
        answer_starts_offsets = []
        temp_tokenized_paragraph = []  # Temporarily used to calculated the amount of tokens in a given paragraph
        offset = 0

        # 1. tokenize all documents separtly and map all answer to current paragrpah and token
        for doc_ind, document in enumerate(context['documents']):
            document['num_of_tokens'] = 0
            document['parts'] = []
            for part_num, part_type, part_text in self.iterate_doc_parts(document):
                part_tokens = self._tokenizer.tokenize(part_text)
                # seems Spacy class is pretty heavy in memory, lets move to a simple representation for now.. 
                part_tokens = [(t.text, t.idx) for t in part_tokens]
                
                document['num_of_tokens'] += len(part_tokens) + 1 # adding 1 for part separator token
                document['parts'].append({'part':part_type,'text':part_text,'tokens':part_tokens})
                
                # computing token_answer_starts (the answer_starts positions in tokens)
                context['qas'] = self.compute_token_answer_starts(context['qas'], doc_ind, part_type, \
                    part_num, part_text, part_tokens)

    def score_documents(self, tokenized_question, documents):
        documents_text = [' '.join([part['text'] for part in doc['parts']]) for doc in documents]
        #documents_text = [doc['text'] for doc in documents]
        tokenized_question_text = [token[0] for token in tokenized_question]
        return self._para_tfidf_scoring.score_paragraphs(tokenized_question_text, documents_text)

    def extract_answers_with_token_idx(self,doc_id,part_ind, answers, part_token_idx_offset):
        answers_list = []
        for answer in answers:
            for alias in answer['aliases']:
                for token_answer_start in alias['token_answer_starts']:
                    if token_answer_start[0] == doc_id and token_answer_start[1] == part_ind:
                        answers_list.append((token_answer_start[2] + part_token_idx_offset, \
                            token_answer_start[3] + part_token_idx_offset, alias['text']))
        return answers_list

    def glue_parts(self, doc_id, document, answers, in_token_idx_char_offest, in_token_idx_offest):
        text = self._PARA_SEP
        tokens = [(self._PARA_SEP.strip(),in_token_idx_char_offest)]
        token_idx_char_offest = in_token_idx_char_offest + len(self._PARA_SEP)
        token_idx_offest = in_token_idx_offest + 1
        norm_answers_list = []

        for part_ind, part in enumerate(document['parts']):
            SEP = self._SEP
            if part['part'] in self._KNOWN_SEP:
                SEP = self._KNOWN_SEP[part['part']]

            # updating text
            text += SEP + part['text'] 
            part_offset = len(SEP) + token_idx_char_offest
            
            # updating tokens
            tokens.append((SEP, token_idx_char_offest))
            part_token_idx_offset = token_idx_offest + 1
            tokens += [(token[0], token[1] + part_offset) for token in part['tokens']]
            
            token_idx_char_offest = in_token_idx_char_offest + len(text)
            token_idx_offest = in_token_idx_offest + len(tokens)
            
            # NOTE we are currently only handling correct answers ... 
            norm_answers_list += self.extract_answers_with_token_idx(doc_id,part_ind, answers,\
                 part_token_idx_offset)
            
        return tokens, text, norm_answers_list, token_idx_char_offest, token_idx_offest

    def sanity_check_answers(self,new_doc):
        updated_answers = copy.deepcopy(new_doc['answers'])
        for answer in new_doc['answers']:
            char_idx_start = new_doc['tokens'][answer[0]][1]
            if answer[1]+1 >= len(new_doc['tokens']):
                char_idx_end = len(new_doc['text'])
            else:
                char_idx_end = new_doc['tokens'][answer[1]+1][1]
            
            self._total_answers += 1    
            if re.match(r'\b{0}\b'.format(re.escape(answer[2])), \
                new_doc['text'][char_idx_start:char_idx_end], re.IGNORECASE) is None:
                if (answer[2].lower() != \
                    new_doc['text'][char_idx_start:char_idx_end].lower()):
                    #print('\nanswer alignment: original: "%s" -VS- found: "%s", removing this answer..' \
                    #    % (answer[2],new_doc['text'][char_idx_start:char_idx_end]))
                    updated_answers.remove(answer)
                    self._answers_removed += 1
        return updated_answers

    def merge_documents(self, documents, qa, ordered_inds): 

        merged_documents = []
        new_doc = {'num_of_tokens':0, 'tokens':[], 'text':'', 'answers':[]}
        curr_doc_ind = 0
        token_idx_char_offest = 0
        token_idx_offest = 0
        for doc_ind in ordered_inds:
            # spliting to new document, Note we assume we are after split documents and each
            # document number of tokens is less than _max_doc_size
            if new_doc['num_of_tokens'] + documents[doc_ind]['num_of_tokens'] > self._max_doc_size:
                # Sanity check: the alias text should be equal the text in answer_start in the paragraph
                # sometimes the original extraction was bad, or the tokenizer makes mistakes... 
                new_doc['answers'] = self.sanity_check_answers(new_doc)

                token_idx_char_offest = 0
                token_idx_offest = 0
                merged_documents.append(new_doc)
                new_doc = {'num_of_tokens':0, 'tokens':[], 'text':'', 'answers':[]}

            tokens, text, norm_answers_list, token_idx_char_offest, token_idx_offest = \
                self.glue_parts(doc_ind, documents[doc_ind], qa['answers'], token_idx_char_offest, token_idx_offest)

            new_doc['num_of_tokens'] += len(tokens)
            new_doc['tokens'] += tokens
            new_doc['text'] += text
            new_doc['answers'] += norm_answers_list

        # adding the remainer document
        if new_doc['num_of_tokens'] > 0:
            new_doc['answers'] = self.sanity_check_answers(new_doc)
            merged_documents.append(new_doc)
        
        return merged_documents

    def filter_paragraphs(self):
        sorted_ix = np.argsort(scores)[:self._max_num_docs]
        # TODO this is ugly
        filtered_paragraphs = []
        filtered_answer_starts_offsets = copy.deepcopy(answer_starts_offsets)
        for new_ind, old_ind in enumerate(sorted_ix):
            filtered_paragraphs.append(paragraphs[old_ind])

            for doc_ind in range(len(answer_starts_offsets)):
                for key in answer_starts_offsets[doc_ind].keys():
                    if answer_starts_offsets[doc_ind][key][0] == old_ind:
                        filtered_answer_starts_offsets[doc_ind][key][0] = new_ind
                    elif answer_starts_offsets[doc_ind][key][0] not in sorted_ix:
                        filtered_answer_starts_offsets[doc_ind][key][0] = -1
        answer_starts_offsets = filtered_answer_starts_offsets
        paragraphs = filtered_paragraphs

        return paragraphs, answer_starts_offsets

    def preprocess(self, contexts):
        
        skipped_qa_count = 0
        all_qa_count = 0

        preprocessed_instances = []

        for context_ind,context in enumerate(contexts):

            self.tokenize_context(context)

            # split paragraphs if needed (assuming paragraphs we merged will not need splitting of course)
            # also the split paragraph will not be merged with a different document. (done after merge)
            self.split_documents(context)   

            # Discarding context that are too long (when len is 0 that means we breaked from context loop)
            # TODO this is mainly relevant when we do not split large paragraphs.
            all_qa_count += len(context['qas'])

            
            # a list of question/answers
            for qa_ind, qa in enumerate(context['qas']): 

                # tokenize question
                tokenized_question = self._tokenizer.tokenize(qa['question'])
                tokenized_question = [(t.text, t.idx) for t in tokenized_question]
        
                # scoring each paragraph for the current question 
                document_scores = self.score_documents(tokenized_question, context['documents'])
                #merged_documents = self.merge_documents(context['documents'], qa, \
                #    np.random.permutation(len(context['documents'])))

                # merge paragraphs if needed until we reach max amount of documents... 
                # (merge is done via tf-idf doc ranking)
                #document_scores = self.score_documents(tokenized_question, merged_documents)
                merged_documents = self.merge_documents(context['documents'], qa, np.argsort(document_scores))

                # filtering the merged documents
                merged_documents = merged_documents[0: self._max_num_docs]

                # Adding Metadata
                metadata = {}
                metadata["question_id"] = qa['id']
                question_text = qa["question"].strip().replace("\n", "")
                answer_texts_list = []
                for answer in qa['answers']:
                    answer_texts_list += [alias['text'] for alias in answer['aliases']]
                metadata["question"] = question_text
                metadata['answer_texts_list'] = answer_texts_list

                # If answer was not found in this question do not yield an instance
                # (This could happen if we used part of the context or in unfiltered context versions)
                if not any(doc['answers']!=[] for doc in merged_documents) and self._require_answer_in_question:
                    skipped_qa_count += 1
                    continue

                for rank, document in enumerate(merged_documents):
                    
                    if document['num_of_tokens'] == 0 or\
                        (self._require_answer_in_doc and document['answers'] == []): 
                        continue

                    metadata['rank'] = rank

                    # adding to cache
                    instance = {'question_text':question_text,
                        'question_tokens':tokenized_question,
                        'metadata':metadata}
                    instance.update(document)
                    preprocessed_instances.append(instance)

        #print("\nFraction of answer that were filtered %f" % (float(self._answers_removed) / self._total_answers))

        return preprocessed_instances, all_qa_count, skipped_qa_count

def _preprocess_t(arg):
    preprocessor = MultiQAPreprocess(*arg[1:7])
    return preprocessor.preprocess(*arg[0:1])

def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]

def group(lst: List[T], max_group_size) -> List[List[T]]:
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst)+max_group_size-1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

def split(lst: List[T], n_groups) -> List[List[T]]:
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parse = argparse.ArgumentParser("Pre-process for DocumentQA/MultiQA model and datareader")
    parse.add_argument("input_file", type=str, help="and input file in MultiQA format (s3 supported)")
    parse.add_argument("output_file", type=str, help="output name and dir to save file (s3 supported)")
    # This is slow, using more processes is recommended
    parse.add_argument("--ndocs", type=int, default=10, help="Number of documents to create")
    parse.add_argument("--docsize", type=int, default=400, help="Max size of each document")
    parse.add_argument("--titles", type=str2bool, default=True, help="Use input documents titles")
    parse.add_argument("--use_rank", type=str2bool, default=True, help="enable sampling")
    parse.add_argument("--require_answer_in_doc", type=str2bool, default=False, help="Add only instance that contain an answer")
    parse.add_argument("--require_answer_in_question", type=str2bool, default=False, help="Add only instance that contain an answer")
    parse.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    parse.add_argument("--sample_size", type=int, default=-1, help="enable sampling")
    
    args = parse.parse_args()
    
    
    # reading file
    contexts = []
    # if `file_path` is a URL, redirect to the cache
    single_file_path = cached_path(args.input_file)
    logger.info("Reading file at %s", args.input_file)

    with zipfile.ZipFile(single_file_path, 'r') as myzip:
        with myzip.open(myzip.namelist()[0]) as myfile:
            dataset_json = json.load(myfile)
        contexts += dataset_json['data']['contexts']

    # sampling
    if args.sample_size > -1:
        random.seed(1)
        contexts = random.sample(contexts, args.sample_size)
    
    if args.n_processes == 1:
        preprocessor = MultiQAPreprocess(args.ndocs, args.docsize, args.titles, args.use_rank, \
             args.require_answer_in_doc,args.require_answer_in_question)
        preprocessed_instances, all_qa_count, skipped_qa_count = preprocessor.preprocess(Tqdm.tqdm(contexts, ncols=80))
    else:
        preprocessed_instances = []
        
        skipped_qa_count = 0
        all_qa_count = 0
        with Pool(args.n_processes) as pool:
            chunks = split(contexts, args.n_processes)
            chunks = flatten_iterable(group(c, 200) for c in chunks)
            pbar = Tqdm.tqdm(total=len(chunks), ncols=80,smoothing=0.0)
            for preproc_inst, all_count, s_count in pool.imap_unordered(_preprocess_t,\
                    [[c, args.ndocs, args.docsize, args.titles, args.use_rank, \
                        args.require_answer_in_doc, args.require_answer_in_question] for c in chunks]):
                preprocessed_instances += preproc_inst 
                all_qa_count += all_count
                skipped_qa_count += s_count
                pbar.update(1)
            pbar.close()

    
    
    # rearranging instances for iterator (we don't use padding noise here, it will be use in the multiqa iterator.)
    ## TODO change question_text to question_tokens
    if True:
        sorting_keys = ['question_tokens','tokens']
        instances_with_lengths = []
        for instance in preprocessed_instances:
            padding_lengths = {key:len(instance[key]) for key in sorting_keys}
            instance_with_lengths = ([padding_lengths[field_name] for field_name in sorting_keys], instance)
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[0])
        preprocessed_instances =  [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]
    

    # saving cache
    print('all_qa_count = %d' % all_qa_count)
    print('skipped_qa_count = %d' % skipped_qa_count)
    preproc_dataset = {'num_examples_used':(all_qa_count - skipped_qa_count, all_qa_count) ,'preprocessed':True,  'preprocessed_instances':preprocessed_instances}
    
    if args.output_file.startswith('s3://'):
        temp_name = 'temp.json.zip'
        output_file = args.output_file.replace('s3://','')
        bucketName = output_file.split('/')[0]
        outPutname = '/'.join(output_file.split('/')[1:])
        with zipfile.ZipFile(temp_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(temp_name, json.dumps(preproc_dataset))
        s3 = boto3.client('s3')
        s3.upload_file(temp_name, bucketName, outPutname, ExtraArgs={'ACL':'public-read'})
    else:
        output_dir = '/'.join(args.output_file.split('/')[0:-1])
        filename = args.output_file.split('/')[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with zipfile.ZipFile(args.output_file, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(filename, json.dumps(preproc_dataset))

if __name__ == "__main__":
    main()


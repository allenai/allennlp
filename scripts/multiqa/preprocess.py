import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy
import sys, os
from typing import TypeVar,Iterable

T = TypeVar('T')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
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
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_context_docs: int = 10,
                 max_context_size: int = 400,
                 save_cache_in_path: string = '',
                 num_of_examples_to_sample: int = None,
                 use_document_titles:bool= False) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_context_docs = max_context_docs
        self._max_context_size = max_context_size
        self._use_document_titles = use_document_titles
        self._num_of_examples_to_sample = num_of_examples_to_sample
        self._para_tfidf_scoring = Paragraph_TfIdf_Scoring(max_context_docs)
        self._save_cache_in_path = save_cache_in_path

    def build_context(self,context):
        # Processing each document separatly
        paragraphs = ['']
        curr_paragraph = 0
        answer_starts_offsets = []
        temp_tokenized_paragraph = []  # Temporarily used to calculated the amount of tokens in a given paragraph
        offset = 0
        for doc_ind, document in enumerate(context['documents']):
            # tokenizing the whole document (title + all snippets concatinated)
            ## TODO add document['rank']
            ## TODO change to <SEP>
            ## TODO handle spliting paragraphs in the middle
            # constracting single context by concatinating parts of the original context
            if self._use_document_titles:
                text_to_add = document['title'] + ' | ' + ' '.join(document['snippets']) + " || "
            else:
                text_to_add = ' '.join(document['snippets']) + " || "

            # Split when number of tokens is larger than _max_context_size.
            tokens_to_add = self._tokenizer.tokenize(text_to_add)
            if len(temp_tokenized_paragraph) + len(tokens_to_add) > self._max_context_size:
                ## Split Paragraphs ##
                temp_tokenized_paragraph = []
                paragraphs.append('')
                curr_paragraph += 1
                # the offset for calculating the answer starts are relative to each paragraphs
                # so for a new paragraph we need to start a new offset.
                offset = 0

            temp_tokenized_paragraph += tokens_to_add
            paragraphs[curr_paragraph] += text_to_add

            # Computing answer_starts offsets:
            if self._use_document_titles:
                answer_starts_offsets.append({'title': [curr_paragraph, offset]})
                offset += len(document['title']) + 3  # we add 3 for the separator ' | '
            else:
                answer_starts_offsets.append({})

            for snippet_ind, snippet in enumerate(document['snippets']):
                answer_starts_offsets[doc_ind][snippet_ind] = [curr_paragraph, offset]
                offset += len(snippet)

                # ' '. adds extra space between the snippets.
                if len(document['snippets']) > 1 and snippet_ind < len(document['snippets']) - 1:
                    offset += 1
            offset += 4  # for " || "

            # offset sanity check
            if offset != len(paragraphs[curr_paragraph]):
                raise ValueError()
        return paragraphs , answer_starts_offsets

    def rank_paragraphs(self, answer_starts_offsets, paragraphs, context):
        tokenized_question = self._tokenizer.tokenize(context['qas'][0]['question'])
        tokenized_question_str = [str(token) for token in tokenized_question]
        scores = self._para_tfidf_scoring.score_paragraphs(tokenized_question_str, paragraphs)
        sorted_ix = np.argsort(scores)[:self._max_context_docs]

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

            paragraphs, answer_starts_offsets = self.build_context(context)

            # scoring each paragraph, and pruning
            paragraphs, answer_starts_offsets = self.rank_paragraphs(answer_starts_offsets, paragraphs, context)

            # Discarding context that are too long (when len is 0 that means we breaked from context loop)
            # TODO this is mainly relevant when we do not split large paragraphs.
            all_qa_count += len(context['qas'])

            # we need to tokenize all the paragraph (again) because previous tokens start the offset count
            # from 0 for each document... # TODO find a better way to do this...
            tokenized_paragraphs = [self._tokenizer.tokenize(paragraph) for paragraph in paragraphs]

            # a list of question/answers
            for qa_ind, qa in enumerate(context['qas']):

                # Adding Metadata
                metadata = {}
                metadata["question_id"] = qa['id']
                question_text = qa["question"].strip().replace("\n", "")
                answer_texts_list = []
                for answer in qa['answers']:
                    answer_texts_list += [alias['text'] for alias in answer['aliases']]
                metadata["question"] = question_text
                metadata['answer_texts_list'] = answer_texts_list


                # calculate new answer starts for the new combined document
                # answer_starts_list is a tuple of (paragraph_number,answer_offset)
                span_starts_list = [{'answers':[],'distractor_answers':[]} for para in paragraphs]
                span_ends_list = [{'answers':[],'distractor_answers':[]} for para in paragraphs]

                if qa['answer_type'] == 'multi_choice':
                    answer_types = ['answers','distractor_answers']
                else:
                    answer_types = ['answers']


                # span_starts_list is a list of dim [answer types] each values is (paragraph num, answer start char offset)
                answer_found = False
                for answer_type in answer_types:
                    for answer in qa[answer_type]:
                        for alias in answer['aliases']:
                            for alias_start in alias['answer_starts']:
                                # It's possible we didn't take all the contexts.
                                if len(answer_starts_offsets) > alias_start[0] and \
                                        alias_start[1] in answer_starts_offsets[alias_start[0]] and  \
                                        (alias_start[1] != 'title' or self._use_document_titles):
                                    answer_start_norm = answer_starts_offsets[alias_start[0]][alias_start[1]][1] + alias_start[2]
                                    answer_start_paragraph = answer_starts_offsets[alias_start[0]][alias_start[1]][0]

                                    # We could have pruned this paragraph
                                    if answer_start_paragraph == -1:
                                        continue

                                    answer_found = True

                                    span_starts_list[answer_start_paragraph][answer_type].append(answer_start_norm)
                                    span_ends_list[answer_start_paragraph][answer_type].append(answer_start_norm + len(alias['text']))


                                    # Sanity check: the alias text should be equal the text in answer_start in the paragraph
                                    x = re.match(r'\b{0}\b'.format(re.escape(alias['text'])),
                                                 paragraphs[answer_start_paragraph][answer_start_norm:answer_start_norm + len(alias['text'])],
                                                 re.IGNORECASE)
                                    if x is None:
                                        if (alias['text'].lower() != paragraphs[answer_start_paragraph][answer_start_norm:answer_start_norm \
                                                                                                     + len(alias['text'])].lower()):
                                            print("answers and paragraph not aligned!")
                                            #raise ValueError("answers and paragraph not aligned!")

                # If answer was not found in this question do not yield an instance
                # (This could happen if we used part of the context or in unfiltered context versions)
                if not answer_found:

                    skipped_qa_count += 1
                    #if context_ind % 30 == 0:
                    #    logger.info('Fraction of QA remaining = %f', ((all_qa_count - skipped_qa_count) / all_qa_count))
                    continue

                for rank, (paragraph,tokenized_paragraph,span_starts,span_ends) in \
                        enumerate(zip(paragraphs,tokenized_paragraphs,span_starts_list,span_ends_list)):

                    if len(tokenized_paragraph) == 0 or span_starts['answers'] == []:
                        continue

                    metadata['rank'] = rank

                    # adding to cache
                    preprocessed_instances.append({'question_text':question_text, 'paragraph':paragraph, 
                                                'span_starts':span_starts, 'span_ends':span_ends, 'metadata':metadata, 'tokenized_paragraph':[(t.text, t.idx) for t in tokenized_paragraph]})

        return preprocessed_instances, all_qa_count, skipped_qa_count

def _preprocess_t(arg):
    preprocessor = MultiQAPreprocess()
    return preprocessor.preprocess(*arg)

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

def main():
    parse = argparse.ArgumentParser("Pre-process for DocumentQA/MultiQA model and datareader")
    parse.add_argument("input_file", type=str, help="and input file in MultiQA format (s3 also supported)")
    parse.add_argument("output_dir", type=str, help="output directory to save file (will be save with the same name)")
    # This is slow, using more processes is recommended
    parse.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    args = parse.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # reading file
    contexts = []
    # if `file_path` is a URL, redirect to the cache
    single_file_path = cached_path(args.input_file)
    logger.info("Reading file at %s", args.input_file)

    with zipfile.ZipFile(single_file_path, 'r') as myzip:
        with myzip.open(myzip.namelist()[0]) as myfile:
            dataset_json = json.load(myfile)
        contexts += dataset_json['data']['contexts']

    #contexts = contexts[0:20000]
    
    if args.n_processes == 1:
        preprocessor = MultiQAPreprocess()
        preprocessed_instances, all_qa_count, skipped_qa_count = preprocessor.preprocess(Tqdm.tqdm(contexts, ncols=80))
    else:
        preprocessed_instances = []
        
        skipped_qa_count = 0
        all_qa_count = 0
        from multiprocessing import Pool
        with Pool(args.n_processes) as pool:
            chunks = split(contexts, args.n_processes)
            chunks = flatten_iterable(group(c, 500) for c in chunks)
            pbar = Tqdm.tqdm(total=len(chunks), ncols=80,smoothing=0.0)
            for preproc_inst,all_count,s_count in pool.imap_unordered(_preprocess_t,[[c] for c in chunks]):
                preprocessed_instances += preproc_inst 
                all_qa_count += all_count
                skipped_qa_count += s_count
                pbar.update(1)
            pbar.close()

    # saving cache
    preproc_dataset = {'num_examples_used':(all_qa_count - skipped_qa_count, all_qa_count) ,'preprocessed':True,  'preprocessed_instances':preprocessed_instances}
    filename = args.input_file.split('/')[-1]
    with zipfile.ZipFile(args.output_dir + '/' + filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(filename, json.dumps(preproc_dataset))

if __name__ == "__main__":
    main()


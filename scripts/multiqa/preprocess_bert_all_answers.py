import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random, math
import sys, os
import boto3
from typing import TypeVar,Iterable
from multiprocessing import Pool
from allennlp.common.elastic_logger import ElasticLogger

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
                 BERT_format,
                 max_context_docs,
                 max_doc_size,
                 use_document_titles,
                 use_rank,
                 require_answer_in_doc,
                 require_answer_in_question,
                 header,
                 DEBUG) -> None:
        self._BERT_format = BERT_format
        self._DEBUG = DEBUG
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
        self._header = header



        if self._BERT_format:
            from pytorch_pretrained_bert.tokenization import BertTokenizer
            # TODO add model type and do_lower_case to the input params
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            self._bert_wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer.tokenize
            self._SEP = ' [SEP] '
            self._KNOWN_SEP = {'rank': ' ', 'title': ' [SEP] '}
            self._bert_do_lowercase = True
            self._never_lowercase = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

            # TODO - allennlp implicetly addes [CLS] and [SEP] tokens in the begining and the end, this
            # shouldn't be the cause in the future...
            self._max_doc_size -= 2

        else:
            # we chose "^..^" because the tokenizer splits the standard "<..>" chars
            self._SEP = ' ^SEP^ '
            self._PARA_SEP = ' ^PARA^ '
            self._KNOWN_SEP = {'rank': ' ', 'title': ' ^TITLE_SEP^ '}

    def wordpiece_tokenizer_len(self, tokens, return_wordpieces = False):
        total_len = 0
        all_word_pieces = []
        for token in tokens:
            # Lowercase if necessary
            if type(token) == tuple:
                text = (token[0].lower()
                        if self._bert_do_lowercase and token[0] not in self._never_lowercase
                        else token[0])
            else:
                text = (token.text.lower()
                        if self._bert_do_lowercase and token.text not in self._never_lowercase
                        else token.text)
            word_pieces = self._bert_wordpiece_tokenizer(text)
            all_word_pieces += word_pieces
            total_len +=  len(word_pieces)
        if return_wordpieces:
            return all_word_pieces
        else:
            return total_len

    def iterate_doc_parts(self,document):
        part_num = 0

        if self._use_rank and 'rank' in document:
            yield (part_num,'rank','^RANK^' + str(document['rank']))
            part_num += 1

        # TODO not using titles creates a crash...
        if self._use_document_titles:
            yield (part_num, 'title',document['title'])
            part_num += 1

        if 'paragraphs' in document:
            for ind,snippet in enumerate(document['paragraphs']):
                # using the special character "«" to indicate newline (new paragraph)
                # so that the spacy tokenizer will split paragraphs with a special token of length 1.
                # (this preserves the all the answer_starts within a snippet...)
                snippet = snippet.replace("\n","«")
                yield (part_num, ind, snippet)
                part_num += 1
        else:
            for ind,snippet in enumerate(document['snippets']):
                snippet = snippet.replace("\n","«")
                yield (part_num, ind, snippet)
                part_num += 1

    def compute_token_answer_starts(self, qas, doc_ind, part_type,part_num, part_text, part_tokens):
        part_offsets = [(token[1], token[1] + len(token[0])) for token in part_tokens]
        for qa in qas:
            for answer in qa['answers']:
                for alias in answer['aliases']:
                    if 'token_answer_starts' not in alias:
                        alias['token_answer_starts'] = []
                    for alias_start in alias['answer_starts']:
                        if alias_start[0] == doc_ind and alias_start[1] == part_type:
                            char_span_start = alias_start[2]
                            char_span_end = char_span_start + len(alias['text'])
                            try:
                                (span_start, span_end), error = util.char_span_to_token_span(part_offsets,
                                                                            (char_span_start, char_span_end))
                                alias['token_answer_starts'].append((doc_ind, part_num,span_start,span_end))
                            except:
                                self._answers_removed += 1
                                self._total_answers += 1
        return qas
    
    def update_answer_docid(self,ans_start_updated_qas, qas, curr_doc_ind, org_doc_ind ,part_num, org_part_num):
        for qa_ind, qa in enumerate(qas):
            for ans_ind, answer in enumerate(qa['answers']):
                for alias_ind, alias in enumerate(answer['aliases']):
                    for ind, alias_start in enumerate(alias['token_answer_starts']):
                        if alias_start[0] == org_doc_ind and alias_start[1] == org_part_num:
                           ans_start_updated_qas[qa_ind]['answers'][ans_ind]['aliases'][alias_ind]['token_answer_starts'][ind] \
                                = (curr_doc_ind, part_num, alias_start[2], alias_start[3])
        return ans_start_updated_qas

    def update_answer_part_split(self,ans_start_updated_qas, qas, org_doc_ind, org_part_ind, new_parts):
        # This is a bit tricky, we are essentially adding new parts so
        # we need to update the answers start for newly added parts (only tokens are important here),
        # and increment the rest of the parts by the amount of newly added parts
        # NOTE we are assuming the title will not be split.

        for qa_ind, qa in enumerate(qas):
            for ans_ind, answer in enumerate(qa['answers']):
                for alias_ind, alias in enumerate(answer['aliases']):
                    inds_not_found = []
                    updated_alias = ans_start_updated_qas[qa_ind]['answers'][ans_ind]['aliases'][alias_ind]
                    for ind, alias_start in enumerate(alias['token_answer_starts']):
                        if alias_start[0] == org_doc_ind and len(updated_alias['token_answer_starts']) > ind:
                           updated_token_start = updated_alias['token_answer_starts'][ind]
                           if alias_start[1] == org_part_ind:
                               for part_ind, new_part in enumerate(new_parts):
                                   if new_part['part_start_org_start'] <= alias_start[2] and \
                                      alias_start[3] < new_part['part_start_org_end']:
                                       ans_start_updated_qas[qa_ind]['answers'][ans_ind]['aliases'][alias_ind]['token_answer_starts'][ind] \
                                           = (org_doc_ind, updated_token_start[1] + part_ind, \
                                              updated_token_start[2] - new_part['part_start_org_start'], \
                                              updated_token_start[3] - new_part['part_start_org_start'])
                                       if self._DEBUG and len(new_part['tokens']) \
                                               < updated_token_start[3] - new_part['part_start_org_start']:
                                           raise (ValueError)
                                       break

                                   # we couldn't find a part that the answer is contained in ... (this is usually
                                   # caused by
                                   if part_ind == len(new_parts) - 1:
                                       inds_not_found.append(ind)

                           elif alias_start[1] > org_part_ind:
                               # increment parts after by 1
                               ans_start_updated_qas[qa_ind]['answers'][ans_ind]['aliases'][alias_ind]['token_answer_starts'][ind] \
                                   = (org_doc_ind, updated_token_start[1] + len(new_parts) - 1, \
                                      updated_token_start[2], updated_token_start[3])

                    # if we couldn't find a part that the answer is contained in, remove it. This is rarely
                    # caused by splitting in the middle of a sentence, like in TriviaQA-web
                    if len(inds_not_found)>0:
                        self._total_answers += len(inds_not_found)
                        self._answers_removed += len(inds_not_found)
                        updated_alias['token_answer_starts'] = [updated_alias['token_answer_starts'][ind] \
                            for ind in range(len(updated_alias['token_answer_starts'])) if ind not in inds_not_found]

        return ans_start_updated_qas

    def split_part(self, part, ans_start_updated_qas, qas, org_doc_ind, org_part_ind, max_question_len):

        # this situation is not ideal for we don't have any good way to split
        # paragraphs. we will try using sentences by utilizing the endline "." token

        # iterating over sentences (end with '.' tokens) + one for the end of the text / part
        new_lines = [(ind, token) for ind, token in enumerate(part['tokens']) if (token[0] == '.' or token[0] == ';')] + \
                        [(len(part['tokens']),('.',len(part['text'])))] * 2

        # in TriviaQA-web even newlines don't always help, in that case well need to just brutally split the part...
        split_points = []
        last_split_token = 0
        ind = 0
        while ind < len(new_lines):
            if new_lines[ind][0] - last_split_token + 2 > self._max_doc_size:
                new_split_point = last_split_token + self._max_doc_size - 2
                split_points.append((new_split_point, part['tokens'][new_split_point]))
                last_split_token = new_split_point
            else:
                split_points.append(new_lines[ind])
                last_split_token = new_lines[ind][0]
                ind += 1

        new_parts = []
        last_split_token = 0
        last_split_char = 0
        for ind, sentence_end in enumerate(split_points):
            if self._BERT_format:
                curr_size = self.wordpiece_tokenizer_len(part['tokens'][last_split_token:sentence_end[0]]) + max_question_len + 1
            else:
                curr_size = sentence_end[0] - last_split_token + 2
            # check if to split + part SEP + _PARA_SEP
            if curr_size > self._max_doc_size or ind  == len(split_points) - 1:
                chosen_splitpoint = split_points[ind-1]
                # splitting the original document
                new_parts.append({'part': part['part'],
                    'part_start_org_start':last_split_token,
                    'part_start_org_end': chosen_splitpoint[0],
                    'text': part['text'][last_split_char:chosen_splitpoint[1][1]],
                    'tokens':[(token[0], token[1] - last_split_char) \
                              for token in part['tokens'][last_split_token:chosen_splitpoint[0]]]})

                if self._BERT_format and self._DEBUG and \
                        self.wordpiece_tokenizer_len(new_parts[-1]['tokens']) + max_question_len > self._max_doc_size:
                        raise (ValueError)
                elif self._DEBUG and len(new_parts[-1]['tokens']) > self._max_doc_size:
                        raise(ValueError)
                last_split_char = chosen_splitpoint[1][1]
                last_split_token = chosen_splitpoint[0]

        #ans_start_updated_qas = self.update_answer_part_split(ans_start_updated_qas, qas, org_doc_ind, org_part_ind, new_parts)

        return ans_start_updated_qas, new_parts
    
    def ensure_parts_size(self, document, context, new_documents, ans_start_updated_qas, org_doc_ind, max_question_len):
        # split parts that are too larger than self._max_doc_size
        sized_parts = []
        part_split_performed = False
        for part_ind, part in enumerate(document['parts']):
            if self._BERT_format:
                # number of wordpieces + part SEP + max question tokens
                part_size = self.wordpiece_tokenizer_len(part['tokens']) + 1 + max_question_len
            else:
                # number of tokens + part SEP + _PARA_SEP
                part_size = len(part['tokens']) + 2

            if part_size > self._max_doc_size:
                ans_start_updated_qas, new_parts = self.split_part(part, ans_start_updated_qas, context['qas'], \
                                                                   org_doc_ind, part_ind, max_question_len)
                sized_parts += new_parts
                part_split_performed = True
            else:
                sized_parts.append(part)
        if part_split_performed:
            context['qas'] = ans_start_updated_qas
            document['parts'] = sized_parts
            ans_start_updated_qas = copy.deepcopy(context['qas'])

        return new_documents, ans_start_updated_qas

    def split_documents(self, context):

        if self._BERT_format:
            # In Bert we need to add the question in the begining of every doc,
            # so we need the longest question and make sure all docs are shorter than  self._max_doc_size - max_question_len
            max_question_len = 0
            for qa in context['qas']:
                question_tokens_len = self.wordpiece_tokenizer_len(self._tokenizer.tokenize(qa['question']))
                if question_tokens_len > max_question_len:
                    max_question_len = question_tokens_len
        else:
            max_question_len = None

        ans_start_updated_qas = copy.deepcopy(context['qas'])
        new_documents = copy.deepcopy(context['documents'])
        for org_doc_ind, document in enumerate(context['documents']):

            # only split documents if total amount of tokens is more than _max_doc_size
            if self._BERT_format:
                # No Paragraph Separator (Done implicitly in AllenNLP wordpiece_indexer)
                # num of tokens + num of separators
                num_of_tokens_and_new_part = document['num_of_tokens'] + len(document['parts']) + max_question_len
            else:
                num_of_tokens_and_new_part = document['num_of_tokens'] + len(document['parts']) + 1 # num of tokens + num of separators  + 1
            if num_of_tokens_and_new_part > self._max_doc_size:
                token_cumsum = 0
                new_document = None

                # ensuring part sizes for curr doc are smaller than _max_doc_size, if not splits them (using '.')
                new_documents, ans_start_updated_qas = \
                    self.ensure_parts_size(document, context, new_documents, ans_start_updated_qas, org_doc_ind ,max_question_len)

                # TODO bug here
                #if self._DEBUG:
                #    self.qas_docs_sanity_check_answers(ans_start_updated_qas,new_documents)

                # now just iterate over parts and combine, until we reach max doc size or num of parts..
                # Note we need to keep the original docs in the same location if possible, to avoid recalculating the
                # qas answer starts...
                for part_ind, part in enumerate(document['parts']):
                    if self._BERT_format:
                        curr_size = token_cumsum + self.wordpiece_tokenizer_len(part['tokens']) + part_ind + 1 + max_question_len
                    else:
                        curr_size = token_cumsum + len(part['tokens']) + part_ind + 2

                    # check if to split (accounting for separators to be added later)
                    if curr_size > self._max_doc_size:
                        # splitting the original document, and keeping it in the same location to avoid qas answer start recalc
                        if not new_document:
                            new_documents[org_doc_ind]['num_of_tokens'] = token_cumsum
                            new_documents[org_doc_ind]['parts'] = document['parts'][0:part_ind]

                        new_document = {'parts':[],'num_of_tokens':0,'org_doc':org_doc_ind}
                        # new documents will be appended to the end of the original document list.
                        new_documents.append(new_document)
                        token_cumsum = 0
                        curr_doc_ind = len(new_documents)-1
                    
                    if new_document:
                        #ans_start_updated_qas = self.update_answer_docid(ans_start_updated_qas, context['qas'], curr_doc_ind, org_doc_ind, \
                        #    len(new_document['parts']), part_ind)
                        new_document['parts'].append(part)
                        if self._BERT_format:
                            new_document['num_of_tokens'] += self.wordpiece_tokenizer_len(part['tokens'])
                        else:
                            new_document['num_of_tokens'] += len(part['tokens'])

                    if self._BERT_format:
                        token_cumsum += self.wordpiece_tokenizer_len(part['tokens'])
                    else:
                        token_cumsum += len(part['tokens'])


        context['documents'] = new_documents
        context['qas'] = ans_start_updated_qas

    def tokenize_context(self, context):
        paragraphs = ['']
        curr_paragraph = 0
        answer_starts_offsets = []
        temp_tokenized_paragraph = []  # Temporarily used to calculated the amount of tokens in a given paragraph
        offset = 0

        # tokenize all documents separably and map all answer to current paragraphs and token
        for doc_ind, document in enumerate(context['documents']):
            document['num_of_tokens'] = 0
            document['parts'] = []
            # document "parts" are title, paragraphs (and paragraphs after split, see split_documents for that)
            for part_num, part_type, part_text in self.iterate_doc_parts(document):
                part_tokens = self._tokenizer.tokenize(part_text)
                # seems Spacy class is pretty heavy in memory, lets move to a simple representation for now.. 
                part_tokens = [(t.text, t.idx) for t in part_tokens]
                if self._BERT_format:
                    document['num_of_tokens'] += self.wordpiece_tokenizer_len(part_tokens) + 1  # adding 1 for part separator token
                else:
                    document['num_of_tokens'] += len(part_tokens) + 1  # adding 1 for part separator token

                document['parts'].append({'part':part_type,'text':part_text,'tokens':part_tokens})

                # computing token_answer_starts (the answer_starts positions in tokens)
                #context['qas'] = self.compute_token_answer_starts(context['qas'], doc_ind, part_type, \
                #                                                  part_num, part_text, part_tokens)

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
        if self._BERT_format:
            text = ''
            tokens = []
            token_idx_char_offest = in_token_idx_char_offest
            token_idx_offest = in_token_idx_offest
        else:
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
            tokens.append((SEP.strip(), token_idx_char_offest))
            part_token_idx_offset = token_idx_offest + 1
            tokens += [(token[0], token[1] + part_offset) for token in part['tokens']]
            
            token_idx_char_offest = in_token_idx_char_offest + len(text)
            token_idx_offest = in_token_idx_offest + len(tokens)
            
            # NOTE we are currently only handling correct answers ... 
            norm_answers_list += self.extract_answers_with_token_idx(doc_id, part_ind, answers,\
                 part_token_idx_offset)
            
        return tokens, text, norm_answers_list, token_idx_char_offest, token_idx_offest

    def qas_docs_sanity_check_answers(self, qas, docs):
        for qa_ind, qa in enumerate(qas):
            for ans_ind, answer in enumerate(qa['answers']):
                for alias_ind, alias in enumerate(answer['aliases']):
                    for ind, alias_start in enumerate(alias['token_answer_starts']):
                        if len(docs[alias_start[0]]['parts'][alias_start[1]]['tokens']) < alias_start[3]:
                            raise(ValueError)

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
                if (answer[2].lower().strip() != \
                    new_doc['text'][char_idx_start:char_idx_end].lower().strip()):
                    if self._DEBUG:
                        print('\nanswer alignment: original: "%s" -VS- found: "%s", removing this answer..' \
                        % (answer[2],new_doc['text'][char_idx_start:char_idx_end]))
                    #updated_answers.remove(answer)
                    # We count the cases in which we remove answers, it usually results in a very small amount under 0.005 of
                    # all the answer starts...
                    #self._answers_removed += 1
        return updated_answers

    def create_new_doc(self, qa, ):
        if self._BERT_format:
            char_offset = 0  # accounting for the new [CLS] + space
            tokens = []
            text = ''
            # allennlp implicitly adds this
            #tokens = [('[CLS]', 0)]
            #text = '[CLS] '
            for t in qa['tokenized_question']:
                tokens.append((t[0], t[1] + char_offset))
            text += qa['question'] + ' '
            token_idx_char_offest = len(text)
            token_idx_offest = len(tokens)
            new_doc = {'num_of_tokens': self.wordpiece_tokenizer_len(tokens), \
                       'tokens': tokens, 'text': text, 'answers': []}
        else:
            token_idx_char_offest = 0
            token_idx_offest = 0
            new_doc = {'num_of_tokens':0, 'tokens':[], 'text':'', 'answers':[]}
        return new_doc, token_idx_offest, token_idx_char_offest

    def merge_documents(self, documents, qa, ordered_inds):

        merged_documents = []
        new_doc, token_idx_offest, token_idx_char_offest = self.create_new_doc(qa)


        for doc_ind in ordered_inds:
            # spliting to new document, Note we assume we are after split documents and each
            # document number of tokens is less than _max_doc_size. (Accounting for separators as well)
            if self._BERT_format:
                curr_num_of_tokens = new_doc['num_of_tokens'] + documents[doc_ind]['num_of_tokens'] \
                                     + len(documents[doc_ind]['parts'])
            else:
                curr_num_of_tokens = new_doc['num_of_tokens'] + documents[doc_ind]['num_of_tokens'] \
                    + len(documents[doc_ind]['parts']) + 1
            if  curr_num_of_tokens > self._max_doc_size:
                # Sanity check: the alias text should be equal the text in answer_start in the paragraph
                # sometimes the original extraction was bad, or the tokenizer makes mistakes... 
                new_doc['answers'] = self.sanity_check_answers(new_doc)

                # AllenNLP implicitly adds this...
                #if self._BERT_format:
                #    new_doc['tokens'] += [('[SEP]',token_idx_char_offest + 1)]
                #    new_doc['text'] += ' [SEP]'
                #    new_doc['num_of_tokens'] += 1

                # BERT wordpiece_tokens sanity check
                if self._DEBUG and self._BERT_format and \
                        (self.wordpiece_tokenizer_len(new_doc['tokens']) + 2 > 512 or \
                        self.wordpiece_tokenizer_len(new_doc['tokens']) != new_doc['num_of_tokens']):
                    raise ValueError()


                merged_documents.append(new_doc)
                new_doc, token_idx_offest, token_idx_char_offest = self.create_new_doc(qa)

            tokens, text, norm_answers_list, token_idx_char_offest, token_idx_offest = \
                self.glue_parts(doc_ind, documents[doc_ind], qa['answers'], token_idx_char_offest, token_idx_offest)

            if self._BERT_format:
                new_doc['num_of_tokens'] += self.wordpiece_tokenizer_len(tokens)
            else:
                new_doc['num_of_tokens'] += len(tokens)
            new_doc['tokens'] += tokens
            new_doc['text'] += text
            new_doc['answers'] += norm_answers_list

            if self._DEBUG and new_doc['num_of_tokens'] > self._max_doc_size:
                raise(ValueError)

        # adding the remainer document
        if new_doc['num_of_tokens'] > 0:
            # AllenNLP implicitly adds this...
            #if self._BERT_format:
            #    new_doc['tokens'] += [('[SEP]', token_idx_char_offest + 1)]
            #    new_doc['text'] += ' [SEP]'
            #    new_doc['num_of_tokens'] += 1

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

    def any_found(self, para, answer):
        strip_chars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’´`_'
        # Normalize the paragraph

        words = [w.lower().strip(strip_chars) for w in para]
        occurances = []
        # Locations where the first word occurs
        word_starts = [i for i, w in enumerate(words) if answer[0] == w]
        n_tokens = len(answer)
        #print(word_starts)
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
                occurances.append((start, end))
        return list(set(occurances))

    def find_answers(self, qa, documents):
        strip_chars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’´`_'
        for answer in qa['answers']:
            for alias in answer['aliases']:
                alias['token_answer_starts'] = []
                if len(alias['text']) > 0:
                    tokenized_alias = self._tokenizer.tokenize(alias['text'])
                    answer_tokens = [t.text.lower().strip(strip_chars) for t in tokenized_alias]
                    for doc_ind, doc in enumerate(documents):
                        for part_num, part in enumerate(doc['parts']):
                            part_tokens = [t[0] for t in part['tokens']]
                            occurances = self.any_found(part_tokens, answer_tokens)
                            for instance in occurances:
                                pass
                                alias['token_answer_starts'].append((doc_ind, part_num, instance[0], instance[1]-1))

    def preprocess(self, contexts):
        
        skipped_qa_count = 0
        all_qa_count = 0

        preprocessed_instances = []

        for context_ind, context in enumerate(contexts):

            self.tokenize_context(context)

            # split paragraphs if needed (assuming paragraphs we merged will not need splitting of course)
            # also the split paragraph will not be merged with a different document. (done after merge)
            self.split_documents(context)   

            all_qa_count += len(context['qas'])
            
            # a list of question/answers
            for qa_ind, qa in enumerate(context['qas']): 

                # tokenize question
                tokenized_question = self._tokenizer.tokenize(qa['question'])
                tokenized_question = [(t.text, t.idx) for t in tokenized_question]
                qa['tokenized_question'] = tokenized_question


        
                # scoring each paragraph for the current question 
                document_scores = self.score_documents(tokenized_question, context['documents'])
                #merged_documents = self.merge_documents(context['documents'], qa, \
                #    np.random.permutation(len(context['documents'])))

                # In this version we re find the answer inside the documents:
                self.find_answers(qa, context['documents'])

                # merge paragraphs if needed until we reach max amount of documents... 
                # (merge is done via tf-idf doc ranking)
                #document_scores = self.score_documents(tokenized_question, merged_documents)
                merged_documents = self.merge_documents(context['documents'], qa, np.argsort(document_scores))
                if self._DEBUG and len([doc for doc in context['documents'] if doc['num_of_tokens'] > self._max_doc_size]) > 0:
                    raise(ValueError)

                # filtering the merged documents
                merged_documents = merged_documents[0: self._max_num_docs]
                if self._DEBUG and len([doc for doc in merged_documents if doc['num_of_tokens'] > self._max_doc_size]) > 0:
                    raise(ValueError)

                # Adding Metadata
                qa_metadata = {}
                qa_metadata['dataset'] = self._header['dataset']
                qa_metadata["context_id"] = context['id']
                qa_metadata["question_id"] = qa['id']
                qa_metadata["all_answers"] = qa['answers']
                question_text = qa["question"].strip().replace("\n", "")
                answer_texts_list = []
                for answer in qa['answers']:
                    answer_texts_list += [alias['text'] for alias in answer['aliases']]
                qa_metadata["question"] = question_text
                qa_metadata['answer_texts_list'] = answer_texts_list

                # If answer was not found in this question do not yield an instance
                # (This could happen if we used part of the context or in unfiltered context versions)
                if not any(doc['answers']!=[] for doc in merged_documents) and self._require_answer_in_question:
                    skipped_qa_count += 1
                    continue

                for rank, document in enumerate(merged_documents):
                    
                    if document['num_of_tokens'] == 0 or\
                        (self._require_answer_in_doc and document['answers'] == []): 
                        continue

                    inst_metatdata = copy.deepcopy(qa_metadata)
                    inst_metatdata['rank'] = rank
                    inst_metatdata["instance_id"] = qa['id'] + '-' + str(rank)
                    inst_metatdata['has_answer'] = document['answers'] != []
                    # adding to cache
                    instance = {'question_text':question_text,
                        'question_tokens':tokenized_question,
                        'metadata':inst_metatdata}
                    instance.update(document)
                    preprocessed_instances.append(instance)

        if self._DEBUG:
            print("\nFraction of answer that were filtered %f" % (float(self._answers_removed) / self._total_answers))

        return preprocessed_instances, all_qa_count, skipped_qa_count

def _preprocess_t(arg):
    preprocessor = MultiQAPreprocess(*arg[1:10])
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
    parse.add_argument("--BERT_format", type=str2bool, default=False, help="Output will be in BERT format")
    parse.add_argument("--ndocs", type=int, default=10, help="Number of documents to create")
    parse.add_argument("--docsize", type=int, default=400, help="Max size of each document")
    parse.add_argument("--titles", type=str2bool, default=True, help="Use input documents titles")
    parse.add_argument("--use_rank", type=str2bool, default=True, help="enable sampling")
    parse.add_argument("--require_answer_in_doc", type=str2bool, default=False, help="Add only instance that contain an answer")
    parse.add_argument("--require_answer_in_question", type=str2bool, default=False, help="Add only instance that contain an answer")
    parse.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    parse.add_argument("--sample_size", type=int, default=-1, help="enable sampling")
    parse.add_argument("--MRQA_style", type=str2bool, default=False, help="MRQA output style")
    parse.add_argument("--sort_by_question", type=str2bool, default=False, help="sort by question token length to optimize GPU zero padding.")
    parse.add_argument("--DEBUG", type=str2bool, default=False, help="sort by question token length to optimize GPU zero padding.")
    parse.add_argument("--START_OFFSET", type=int, default=None, help="start from a certain input index")

    args = parse.parse_args()
    
    
    # reading file
    contexts = []
    # if `file_path` is a URL, redirect to the cache
    single_file_path = cached_path(args.input_file)
    logger.info("Reading file at %s", args.input_file)



    with zipfile.ZipFile(single_file_path, 'r') as myzip:
        if myzip.namelist()[0].find('jsonl')>0:
            contexts = []
            with myzip.open(myzip.namelist()[0]) as myfile:
                header = json.loads(myfile.readline())['header']
                for example in myfile:
                    contexts.append(json.loads(example))
        else:
            with myzip.open(myzip.namelist()[0]) as myfile:
                dataset_json = json.load(myfile)
            contexts += dataset_json['data']['contexts']

    # sampling
    if args.sample_size > -1:
        random.seed(2)
        if args.sample_size < len(contexts):
            contexts = random.sample(contexts, args.sample_size)
        else:
            contexts = random.sample(contexts,len(contexts))

        sampled_contexts = []
        num_of_qas = 0
        for context in contexts:
            if num_of_qas > args.sample_size:
                break
            sampled_contexts.append(context)
            num_of_qas += len(context['qas'])
        contexts = sampled_contexts

    if args.DEBUG and args.START_OFFSET is not None:
        contexts = contexts[args.START_OFFSET:]

    if args.n_processes == 1:
        preprocessor = MultiQAPreprocess(args.BERT_format,args.ndocs, args.docsize, args.titles, args.use_rank, \
             args.require_answer_in_doc,args.require_answer_in_question, header, args.DEBUG)
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
                    [[c, args.BERT_format,args.ndocs, args.docsize, args.titles, args.use_rank, \
                        args.require_answer_in_doc, args.require_answer_in_question, header, args.DEBUG] for c in chunks]):
                preprocessed_instances += preproc_inst 
                all_qa_count += all_count
                skipped_qa_count += s_count
                pbar.update(1)
            pbar.close()

    
    
    # rearranging instances for iterator (we don't use padding noise here, it will be use in the multiqa iterator.)
    ## TODO change question_text to question_tokens
    if args.sort_by_question:
        # bucketing by QuestionID
        instance_list = preprocessed_instances
        instance_list = sorted(instance_list, key=lambda x: x['metadata']['question_id'])
        intances_question_id = [instance['metadata']['question_id'] for instance in instance_list]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                                  range(len(split_inds) - 1)]
        print('num of per_question_instances = %d' % (len(per_question_instances)))

        # sorting
        sorting_keys = ['question_tokens', 'tokens']
        instances_with_lengths = []
        for instance in per_question_instances:
            padding_lengths = {key: len(instance[0][key]) for key in sorting_keys}
            instance_with_lengths = ([padding_lengths[field_name] for field_name in sorting_keys], instance)
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[0])
        per_question_instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]
        preprocessed_instances = []
        for question_instances in per_question_instances:
            preprocessed_instances += question_instances


    # saving cache
    print('all_qa_count = %d' % all_qa_count)
    print('skipped_qa_count = %d' % skipped_qa_count)
    preproc_dataset = {'num_examples_used':(all_qa_count - skipped_qa_count, all_qa_count) ,'preprocessed':True,  'preprocessed_instances':preprocessed_instances}

    # building header
    instance_with_answers = len([1 for instance in preprocessed_instances if instance['metadata']['has_answer']])
    preproc_header = {'preproc.num_of_documents':args.ndocs,
              'preproc.max_doc_size':args.docsize,
              'preproc.use_titles':args.titles,
              'preproc.use_rank':args.use_rank,
              'preproc.require_answer_in_doc': args.require_answer_in_doc,
              'preproc.require_answer_in_question': args.require_answer_in_question,
              'preproc.total_num_of_questions': all_qa_count,
              'preproc.num_of_questions_used': all_qa_count - skipped_qa_count,
              'preproc.frac_of_instances_with_answers':float(instance_with_answers) / len(preprocessed_instances),
              'preproc.num_of_instances':len(preprocessed_instances),
              'preproc.final_qas_used_fraction': \
                    header['qas_used_fraction'] * (all_qa_count - skipped_qa_count) / all_qa_count}
    preproc_header.update(header)

    if args.MRQA_style:
        # changing the header...
        preproc_header = {'header':{'dataset':header['dataset'],
                                    'split': header['split_type'],
                                    '#examples':len(preprocessed_instances)}}
        mrqa_foramt_instances = []
        for instance in preprocessed_instances:
            answers = instance['metadata']['all_answers']

            ## NOTE!! we are supporting only 1 answer here ...
            new_answers = {"answer":answers[0]["answer"],"aliases":[]}

            for alias in answers[0]['aliases']:
                for answer_in_tokens in instance['answers']:
                    if alias['text'] == answer_in_tokens[2]:
                        new_answers["aliases"].append({"answer_starts":instance['tokens'][answer_in_tokens[0]][1], \
                                                           "text":alias['text']})

            mrqa_instance = {'id':instance['metadata']['question_id'],
                             "context": instance['text'],
                             "question": instance['question_text'],
                             "answer": new_answers}

            mrqa_foramt_instances.append(mrqa_instance)

        preprocessed_instances = mrqa_foramt_instances


    if args.output_file.startswith('s3://'):
        output_file = args.output_file.replace('s3://','')
        bucketName = output_file.split('/')[0]
        outPutname = '/'.join(output_file.split('/')[1:])
        local_filename = outPutname.replace('/','_')
        with open(local_filename.replace('.zip',''), "w") as f:
            # first JSON line is header
            f.write(json.dumps({'header':preproc_header}) + '\n')
            for instance in preprocessed_instances:
                f.write(json.dumps(instance, sort_keys=True, indent=4 ) + '\n')

        with zipfile.ZipFile(local_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(local_filename.replace('.zip',''))

        s3 = boto3.client('s3')
        s3.upload_file(local_filename , bucketName, outPutname, ExtraArgs={'ACL':'public-read'})

        os.remove(local_filename)
        os.remove(local_filename.replace('.zip',''))
    else:
        output_dir = '/'.join(args.output_file.split('/')[0:-1])
        dataset = args.output_file.split('/')[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_file.replace('.zip',''), "w") as f:
            # first JSON line is header
            f.write(json.dumps({'header': preproc_header}) + '\n')
            for instance in preprocessed_instances:
                if False:
                    s = json.dumps(instance, sort_keys=True, indent=4)
                    # just making the answer starts in the sample no have a newline for every offset..
                    s = re.sub('\n\s*(\d+)', r'\1', s)
                    s = re.sub('\n\s*"title"', r'"title"', s)
                    s = re.sub('(\d+)\n\s*]', r'\1]', s)
                    s = re.sub('(\d+)],\n\s*', r'\1],', s)
                    s = re.sub('\[\s*\n', r'[', s)
                    s = re.sub('\[\s*', r'[', s)
                    f.write(s + '\n')
                else:
                    f.write(json.dumps(instance) + '\n')

        with zipfile.ZipFile(args.output_file, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(args.output_file.replace('.zip',''))

        os.remove(args.output_file.replace('.zip',''))

    ElasticLogger().write_log('INFO', 'Dataset Preproc Stats', context_dict=preproc_header)


if __name__ == "__main__":
    main()


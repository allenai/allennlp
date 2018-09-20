import re
import itertools
import nltk
import json
import csv
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple, Union, Set

from overrides import overrides
from unidecode import unidecode

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

# == stop words that will be omitted by ContextGenerator
STOP_WORDS = ["", "",  "all", "being", "-", "over", "through", "yourselves", "its", "before", 
             "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do", 
             "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him", "nor", 
             "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are", 
             "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?", 
             "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn", 
             "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your", 
             "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves", 
             ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he", 
             "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then", 
             "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!", "again", "'ll", 
            "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan", "'t", "'s", "our", "after", 
            "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so", "the", "having", "once"]

                                 

class ContextGenerator:
    """
    A Barebones implementation similar to https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/table/wtq/preprocess.py
    for extracting entities from a question given a table. 
    """

    def __init__(self, 
                 cell_values : Set[str], 
                 column_type_statistics : List[Dict[str,int]],
                 column_index_to_name : Dict[int,str]) -> None:
        self.cell_values = cell_values
        self.column_types = { column_index_to_name[column_index] : max(column_type_statistics[column_index]) for column_index in column_index_to_name }
            
       

    @classmethod
    def parse_file(cls, filename: str, question: List[Token], max_tokens_for_num_cell : int) -> 'TableQuestionKnowledgeGraph':
        with open(filename, 'r') as file_pointer:
        	reader = csv.reader(file_pointer,  delimiter='\t', quoting=csv.QUOTE_NONE)
        	# obtain column information
        	lines = [line for line in reader]
        	column_index_to_name = {}

        	header = lines[0] # the first line is the header
        	index = 1
        	while lines[index][0] == '-1':
        		# column names start with fb:row.row. 
        		curr_line = lines[index]
        		column_name_sempre = curr_line[2]
        		column_index = curr_line[1]
        		column_name = column_name_sempre.replace('fb:row.row.', '')
        		column_index_to_name[column_index] = column_name
            	index += 1

        
        	column_node_type_info = [{'string' : 0, 'number' : 0, 'date' : 0} 
                                 for col in column_index_to_name]
        	cell_values = set()
        	while index < len(lines):
        		curr_line = lines[index]
            	column_index = curr_line[1]
            	node_info = dict(zip(header, curr_line))
                cell_values.add(cls._normalize_string(node_info['content']))  
            	if node_info['date']:
            		column_node_type_info[column_index]['date'] += 1 
            	# If cell contains too many tokens, then likely not number
            	elif node_info['number'] and num_tokens < max_tokens_for_num_cell:
            		column_node_type_info[column_index]['number'] += 1 
            	else:
            		column_node_type_info[column_index]['string'] += 1


        	return cls(cell_values, column_node_type_info, column_index_to_name)


    def get_entities_from_question(self, question : List[str]):
        entity_data = []
        for i, token in enumerate(question):
            if token in STOP_WORDS: continue
            normalized_token = self._normalize_string(token)
            if len(normalized_token) == 0: continue
            if self._string_in_table(normalized_token):
                curr_data = {'value' : normalized_token, 'token_start' : i, 'token_end' : i+1}
                entity_data.append(curr_data)
                                  

        expanded_entities = self._expand_entities(question, entity_data, table)
        return expanded_entities #TODO(shikhar) Handle conjunctions
    
        

    def _string_in_table(self, candidate : str) -> bool:
        for cell_value in self.cell_values:
            if candidate in cell_value: return True
        return False


    def _process_conjunction(self, entity_data):
        raise NotImplementedError

    def _expand_entities(self, question, entity_data):
        new_ents = []
        for ent in entity_data:
            # to ensure the same strings are not used over and over
            if new_ents and ent['token_end'] <= new_ents[-1]['token_end']: continue
            curr_st = ent['token_start']
            curr_en = ent['token_end']
            curr_tok = ent['value']

            while curr_en < len(question): 
                next_tok = question[curr_en]
                next_tok_normalized = self._normalize_string(next_tok) 
                candidate = "%s_%s" %(curr_tok, next_tok_normalized) 
                if self._string_in_table(candidate):
                    curr_en += 1
                    curr_tok = candidate
                else: 
                    break

            new_ents.append({'token_start' : curr_st, 'token_end' : curr_en, 'value' : curr_tok})

        return new_ents

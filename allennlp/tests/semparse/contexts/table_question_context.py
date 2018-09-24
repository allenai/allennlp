# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.semparse.contexts import TableQuestionContext

import glob
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

tests = [ {"strings": ["film"], "numbers": [["1", "first"], ["1943", "1943"]], "question": "what was the first tamil-language film in 1943?"},
        {"question": "on what date did the eagles score the least points?"},
        {"strings": ["france", "south_korea"], "question": "was the positioning better the year of the france venue or the year of the south korea venue?"},
        {"strings": ["name"], "numbers": [["5", "five"]],  "question": "name all the nations that won at least five silver medals."},
        {"strings": ["r_beitzel"], "question": "what is weldell r. beitzel's party?"},
        {"strings": ["san", "timbers"], "numbers": [["1979", "1979"]],  "question": "how many different players scored for the san jose earthquakes during their 1979 home opener against the timbers?"},
        {"numbers": [["191617", "191617"], ["100", "100"]],  "question": "how many players on the 191617 illinois fighting illini men's basketball team had more than 100 points scored?"},
        {"strings": ["matt_kenset"], "numbers": [["2", "february"], ["26", "26"], ["2006", "2006"]], "question": "how many laps did matt kenset complete on february 26, 2006."},
        {"strings": ["took_office"], "question": "which elected successor took office the earliest?"},
        {"strings": ["cherry"], "question": "who is ranked previous to don cherry?"} ]

class TestTableQuestionContext(AllenNlpTestCase):
    def test_extract_entities_from_question(self):
        question = "did the bell system strike last longer in 1971 or 1983?"   
        # Read sampled tests and see if our model can also recover these  
 
    def test_read_from_file(self):
        tokenizer =  WordTokenizer(SpacyWordSplitter(pos_tags=True))
        for table in glob.glob('test_tables/*'):
            table_question_context = TableQuestionContext.read_from_lines(table)
            # TODO (smurty) make this less ugly
            test_id = table.replace('TEST-', '').replace('.table', '')
            curr_test = tests[int(test_id)]
            curr_question = curr_test['question']
            str_entities  = curr_test['strings'] if strings in curr_test else []
            num_entities  = curr_test['numbers'] if numbers in curr_test else []   
            str_entities_predicted, num_entities_predicted = table_question_context.get_entities_from_question(tokenizer.tokenize(curr_question.lower()))
            assert str_entities_predicted == str_entities
            assert num_entities_predicted == num_entities


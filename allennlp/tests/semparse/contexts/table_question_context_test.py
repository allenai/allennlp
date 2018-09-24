# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.semparse.contexts import TableQuestionContext

import glob
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

MAX_TOKENS_FOR_NUMBER = 6

class TestTableQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True)) 

    def test_number_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-7.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        question = "how many players on the 191617 illinois fighting illini men's basketball team had more than 100 points scored?"
        _, number_entities = table_question_context.get_entities_from_question(self.tokenizer.tokenize(question))
        assert number_entities == [("191617", "191617"), ("100", "100")]
    
    def test_date_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-8.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        question = "how many laps did matt kenset complete on february 26, 2006."
        _, number_entities = table_question_context.get_entities_from_question(self.tokenizer.tokenize(question))
        assert number_entities == [("2", "february"), ("26", "26"), ("2006", "2006")]  
         
    def test_date_extraction_2(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-6.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        question = "how many different players scored for the san jose earthquakes during their 1979 home opener against the timbers?"
        _, number_entities = table_question_context.get_entities_from_question(self.tokenizer.tokenize(question))
        assert number_entities[0] == ("1979", "1979") 

    def test_multiword_entity_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-3.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        question = "was the positioning better the year of the france venue or the year of the south korea venue?"  
        entities, _ = table_question_context.get_entities_from_question(self.tokenizer.tokenize(question))
        assert entities == ["france", "south_korea"]  

    def test_rank_number_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-1.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        question = "what was the first tamil-language film in 1943?"  
        _, numbers = table_question_context.get_entities_from_question(self.tokenizer.tokenize(question))
        assert numbers[0] == ("1", "first")

    def test_null_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-2.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        question = "on what date did the eagles score the least points?"
        entities, numbers = table_question_context.get_entities_from_question(self.tokenizer.tokenize(question))
        assert entities == []
        assert numbers == []  

    def test_numerical_column_type_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-7.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        predicted_types = table_question_context.column_types
        assert predicted_types["games_played"] == "number"
        assert predicted_types["field_goals"]  == "number"
        assert predicted_types["free_throws"] == "number"
        assert predicted_types["points"] == "number"

    def test_date_column_type_extraction_1(self): 
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-5.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        predicted_types = table_question_context.column_types
        assert predicted_types["first_elected"] == "date"

    def test_date_column_type_extraction_2(self): 
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-9.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        predicted_types = table_question_context.column_types
        assert predicted_types["date_of_appointment"] == "date"
        assert predicted_types["date_of_election"] == "date"

    def test_string_column_types_extraction(self):
        table_question_context = TableQuestionContext.read_from_file('%s/data/corenlp_processed_tables/TEST-10.table' %self.FIXTURES_ROOT, MAX_TOKENS_FOR_NUMBER)
        predicted_types = table_question_context.column_types
        assert predicted_types["birthplace"] == "string"
        assert predicted_types["advocate"] == "string"
        assert predicted_types["notability"] == "string"
        assert predicted_types["name"] == "string"

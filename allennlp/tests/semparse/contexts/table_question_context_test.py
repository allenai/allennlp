# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

MAX_TOKENS_FOR_NUMBER = 6

class TestTableQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

    def test_number_extraction(self):
        question = """how many players on the 191617 illinois fighting illini men's basketball team
                      had more than 100 points scored?"""
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-7.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, number_entities = table_question_context.get_entities_from_question()
        assert number_entities == [("191617", 5), ("100", 16)]

    def test_date_extraction(self):
        question = "how many laps did matt kenset complete on february 26, 2006."
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-8.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, number_entities = table_question_context.get_entities_from_question()
        assert number_entities == [("2", 8), ("26", 9), ("2006", 11)]

    def test_date_extraction_2(self):
        question = """how many different players scored for the san jose earthquakes during their
                      1979 home opener against the timbers?"""
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-6.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, number_entities = table_question_context.get_entities_from_question()
        assert number_entities == [("1979", 12)]

    def test_multiword_entity_extraction(self):
        question = "was the positioning better the year of the france venue or the year of the south korea venue?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-3.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        entities, _ = table_question_context.get_entities_from_question()
        assert entities == ["france", "south_korea"]

    def test_rank_number_extraction(self):
        question = "what was the first tamil-language film in 1943?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-1.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, numbers = table_question_context.get_entities_from_question()
        assert numbers == [("1", 3), ('1943', 9)]

    def test_null_extraction(self):
        question = "on what date did the eagles score the least points?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-2.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        entities, numbers = table_question_context.get_entities_from_question()
        # "Eagles" does not appear in the table.
        assert entities == []
        assert numbers == []

    def test_numerical_column_type_extraction(self):
        question = """how many players on the 191617 illinois fighting illini men's basketball team
                      had more than 100 points scored?"""
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-7.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        predicted_types = table_question_context.column_types
        assert predicted_types["games_played"] == "number"
        assert predicted_types["field_goals"] == "number"
        assert predicted_types["free_throws"] == "number"
        assert predicted_types["points"] == "number"

    def test_date_column_type_extraction_1(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-5.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        predicted_types = table_question_context.column_types
        assert predicted_types["first_elected"] == "date"

    def test_date_column_type_extraction_2(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-9.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        predicted_types = table_question_context.column_types
        assert predicted_types["date_of_appointment"] == "date"
        assert predicted_types["date_of_election"] == "date"

    def test_string_column_types_extraction(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-10.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        predicted_types = table_question_context.column_types
        assert predicted_types["birthplace"] == "string"
        assert predicted_types["advocate"] == "string"
        assert predicted_types["notability"] == "string"
        assert predicted_types["name"] == "string"

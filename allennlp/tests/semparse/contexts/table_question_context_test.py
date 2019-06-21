# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.contexts.table_question_context import Date
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestTableQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

    def test_table_data(self):
        question = "what was the attendance when usl a league played?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/wikitables/sample_table.tagged'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        assert table_question_context.table_data == [{'date_column:year': Date(2001, -1, -1),
                                                      'number_column:year': 2001.0,
                                                      'string_column:year': '2001',
                                                      'number_column:division': 2.0,
                                                      'string_column:division': '2',
                                                      'string_column:league': 'usl_a_league',
                                                      'string_column:regular_season': '4th_western',
                                                      'number_column:regular_season': 4.0,
                                                      'string_column:playoffs': 'quarterfinals',
                                                      'string_column:open_cup': 'did_not_qualify',
                                                      'number_column:open_cup': None,
                                                      'number_column:avg_attendance': 7169.0,
                                                      'string_column:avg_attendance': '7_169'},
                                                     {'date_column:year': Date(2005, -1, -1),
                                                      'number_column:year': 2005.0,
                                                      'string_column:year': '2005',
                                                      'number_column:division': 2.0,
                                                      'string_column:division': '2',
                                                      'string_column:league': 'usl_first_division',
                                                      'string_column:regular_season': '5th',
                                                      'number_column:regular_season': 5.0,
                                                      'string_column:playoffs': 'quarterfinals',
                                                      'string_column:open_cup': '4th_round',
                                                      'number_column:open_cup': 4.0,
                                                      'number_column:avg_attendance': 6028.0,
                                                      'string_column:avg_attendance': '6_028'}]

    def test_table_data_from_untagged_file(self):
        question = "what was the attendance when usl a league played?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/wikitables/sample_table.tsv'
        table_lines = [line.strip() for line in open(test_file).readlines()]
        table_question_context = TableQuestionContext.read_from_lines(table_lines, question_tokens)
        # The content in the table represented by the untagged file we are reading here is the same as the one we
        # had in the tagged file above, except that we have a "Score" column instead of "Avg. Attendance" column,
        # which is changed to test the num2 extraction logic. I've shown the values not being extracted here as
        # well and commented them out.
        assert table_question_context.table_data == [{'number_column:year': 2001.0,
                                                      # The value extraction logic we have for untagged lines does
                                                      # not extract this value as a date.
                                                      #'date_column:year': Date(2001, -1, -1),
                                                      'string_column:year': '2001',
                                                      'number_column:division': 2.0,
                                                      'string_column:division': '2',
                                                      'string_column:league': 'usl_a_league',
                                                      'string_column:regular_season': '4th_western',
                                                      # We only check for strings that are entirely numbers. So 4.0
                                                      # will not be extracted.
                                                      #'number_column:regular_season': 4.0,
                                                      'string_column:playoffs': 'quarterfinals',
                                                      'string_column:open_cup': 'did_not_qualify',
                                                      #'number_column:open_cup': None,
                                                      'number_column:score': 20.0,
                                                      'num2_column:score': 30.0,
                                                      'string_column:score': '20_30'},
                                                     {'number_column:year': 2005.0,
                                                      #'date_column:year': Date(2005, -1, -1),
                                                      'string_column:year': '2005',
                                                      'number_column:division': 2.0,
                                                      'string_column:division': '2',
                                                      'string_column:league': 'usl_first_division',
                                                      'string_column:regular_season': '5th',
                                                      # Same here as in the "division" column for the first row.
                                                      # 5.0 will not be extracted from "5th".
                                                      #'number_column:regular_season': 5.0,
                                                      'string_column:playoffs': 'quarterfinals',
                                                      'string_column:open_cup': '4th_round',
                                                      #'number_column:open_cup': 4.0,
                                                      'number_column:score': 50.0,
                                                      'num2_column:score': 40.0,
                                                      'string_column:score': '50_40'}]

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
        assert entities == [("string:france", ["string_column:venue"]),
                            ("string:south_korea", ["string_column:venue"])]

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
        column_names = table_question_context.column_names
        assert "number_column:games_played" in column_names
        assert "number_column:field_goals" in column_names
        assert "number_column:free_throws" in column_names
        assert "number_column:points" in column_names

    def test_date_column_type_extraction_1(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-5.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        column_names = table_question_context.column_names
        assert "date_column:first_elected" in column_names

    def test_date_column_type_extraction_2(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-9.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        column_names = table_question_context.column_names
        assert "date_column:date_of_appointment" in column_names
        assert "date_column:date_of_election" in column_names

    def test_string_column_types_extraction(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-10.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        column_names = table_question_context.column_names
        assert "string_column:birthplace" in column_names
        assert "string_column:advocate" in column_names
        assert "string_column:notability" in column_names
        assert "string_column:name" in column_names

    def test_number_and_entity_extraction(self):
        question = "other than m1 how many notations have 1 in them?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f"{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-11.table"
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        string_entities, number_entities = table_question_context.get_entities_from_question()
        assert string_entities == [("string:m1", ["string_column:notation"]),
                                   ("string:1", ["string_column:position"])]
        assert number_entities == [("1", 2), ("1", 7)]

    def test_get_knowledge_graph(self):
        question = "other than m1 how many notations have 1 in them?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f"{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-11.table"
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        knowledge_graph = table_question_context.get_table_knowledge_graph()
        entities = knowledge_graph.entities
        # -1 is not in entities because there are no date columns in the table.
        assert sorted(entities) == ['1',
                                    'number_column:notation',
                                    'number_column:position',
                                    'string:1',
                                    'string:m1',
                                    'string_column:mnemonic',
                                    'string_column:notation',
                                    'string_column:position',
                                    'string_column:short_name',
                                    'string_column:swara']
        neighbors = knowledge_graph.neighbors
        # Each number extracted from the question will have all number and date columns as
        # neighbors. Each string entity extracted from the question will only have the corresponding
        # column as the neighbor.
        neighbors_with_sets = {key: set(value) for key, value in neighbors.items()}
        assert neighbors_with_sets == {'1': {'number_column:position', 'number_column:notation'},
                                       'string_column:mnemonic': set(),
                                       'string_column:short_name': set(),
                                       'string_column:swara': set(),
                                       'number_column:position': {'1'},
                                       'number_column:notation': {'1'},
                                       'string:m1': {'string_column:notation'},
                                       'string:1': {'string_column:position'},
                                       'string_column:notation': {'string:m1'},
                                       'string_column:position': {'string:1'}}
        entity_text = knowledge_graph.entity_text
        assert entity_text == {'1': '1',
                               'string:m1': 'm1',
                               'string:1': '1',
                               'string_column:notation': 'notation',
                               'number_column:notation': 'notation',
                               'string_column:mnemonic': 'mnemonic',
                               'string_column:short_name': 'short name',
                               'string_column:swara': 'swara',
                               'number_column:position': 'position',
                               'string_column:position': 'position'}


    def test_knowledge_graph_has_correct_neighbors(self):
        question = "when was the attendance greater than 5000?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/wikitables/sample_table.tagged'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        knowledge_graph = table_question_context.get_table_knowledge_graph()
        neighbors = knowledge_graph.neighbors
        # '5000' is neighbors with number and date columns. '-1' is in entities because there is a
        # date column, which is its only neighbor.
        assert set(neighbors.keys()) == {'date_column:year',
                                         'number_column:year',
                                         'string_column:year',
                                         'number_column:division',
                                         'string_column:division',
                                         'string_column:league',
                                         'string_column:regular_season',
                                         'number_column:regular_season',
                                         'string_column:playoffs',
                                         'string_column:open_cup',
                                         'number_column:open_cup',
                                         'number_column:avg_attendance',
                                         'string_column:avg_attendance',
                                         '5000',
                                         '-1'}
        assert set(neighbors['date_column:year']) == {'5000', '-1'}
        assert neighbors['number_column:year'] == ['5000']
        assert neighbors['string_column:year'] == []
        assert neighbors['number_column:division'] == ['5000']
        assert neighbors['string_column:division'] == []
        assert neighbors['string_column:league'] == []
        assert neighbors['string_column:regular_season'] == []
        assert neighbors['number_column:regular_season'] == ['5000']
        assert neighbors['string_column:playoffs'] == []
        assert neighbors['string_column:open_cup'] == []
        assert neighbors['number_column:open_cup'] == ['5000']
        assert neighbors['number_column:avg_attendance'] == ['5000']
        assert neighbors['string_column:avg_attendance'] == []
        assert set(neighbors['5000']) == {'date_column:year',
                                          'number_column:year',
                                          'number_column:division',
                                          'number_column:avg_attendance',
                                          'number_column:regular_season',
                                          'number_column:open_cup'}
        assert neighbors['-1'] == ['date_column:year']

# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List
from functools import partial

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.type_declarations import wikitables_variable_free as types
from allennlp.semparse.worlds import WikiTablesVariableFreeWorld
from allennlp.semparse import ParsingError


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestWikiTablesVariableFreeWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2013', '?']]
        self.table_file = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table.tagged'
        self.table_context = TableQuestionContext.read_from_file(self.table_file, question_tokens)
        self.world_with_2013 = WikiTablesVariableFreeWorld(self.table_context)
        usl_league_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', 'with', 'usl',
                                                'a', 'league', '?']]
        self.world_with_usl_a_league = self._get_world_with_question_tokens(usl_league_tokens)

    def _get_world_with_question_tokens(self, tokens: List[Token]) -> WikiTablesVariableFreeWorld:
        table_context = TableQuestionContext.read_from_file(self.table_file, tokens)
        world = WikiTablesVariableFreeWorld(table_context)
        return world

    def test_get_valid_actions_returns_correct_set(self):
        # This test is long, but worth it.  These are all of the valid actions in the grammar, and
        # we want to be sure they are what we expect.
        valid_actions = self.world_with_2013.get_valid_actions()
        assert set(valid_actions.keys()) == {
                "<r,<g,s>>",
                "<r,<f,<n,r>>>",
                "<r,<c,r>>",
                "<r,<g,r>>",
                "<r,<r,<f,n>>>",
                "<r,<t,<s,r>>>",
                "<n,<n,<n,d>>>",
                "<r,<m,<d,r>>>",
                "<r,<f,n>>",
                "<r,r>",
                "<r,n>",
                "d",
                "n",
                "s",
                "m",
                "t",
                "f",
                "r",
                "@start@",
                }

        check_productions_match(valid_actions['<r,<g,s>>'],
                                ['mode', 'select'])

        check_productions_match(valid_actions['<r,<f,<n,r>>>'],
                                ['filter_number_equals', 'filter_number_greater',
                                 'filter_number_greater_equals', 'filter_number_lesser',
                                 'filter_number_lesser_equals', 'filter_number_not_equals'])

        check_productions_match(valid_actions['<r,<c,r>>'],
                                ['argmax', 'argmin'])

        check_productions_match(valid_actions['<r,<g,r>>'],
                                ['same_as'])

        check_productions_match(valid_actions['<r,<r,<f,n>>>'],
                                ['diff'])

        check_productions_match(valid_actions['<r,<t,<s,r>>>'],
                                ['filter_in', 'filter_not_in'])

        check_productions_match(valid_actions['<n,<n,<n,d>>>'],
                                ['date'])

        check_productions_match(valid_actions['<r,<m,<d,r>>>'],
                                ['filter_date_equals', 'filter_date_greater',
                                 'filter_date_greater_equals', 'filter_date_lesser',
                                 'filter_date_lesser_equals', 'filter_date_not_equals'])

        check_productions_match(valid_actions['<r,<f,n>>'],
                                ['average', 'max', 'min', 'sum'])

        check_productions_match(valid_actions['<r,r>'],
                                ['first', 'last', 'next', 'previous'])

        check_productions_match(valid_actions['<r,n>'],
                                ['count'])

        # These are the columns in table, and are instance specific.
        check_productions_match(valid_actions['m'],
                                ['date_column:year'])

        check_productions_match(valid_actions['f'],
                                ['number_column:avg_attendance',
                                 'number_column:division'])

        check_productions_match(valid_actions['t'],
                                ['string_column:league',
                                 'string_column:playoffs',
                                 'string_column:open_cup',
                                 'string_column:regular_season'])

        check_productions_match(valid_actions['@start@'],
                                ['d', 'n', 's'])

        # The question does not produce any strings. It produces just a number.
        check_productions_match(valid_actions['s'],
                                ['[<r,<g,s>>, r, m]',
                                 '[<r,<g,s>>, r, f]',
                                 '[<r,<g,s>>, r, t]'])

        check_productions_match(valid_actions['d'],
                                ['[<n,<n,<n,d>>>, n, n, n]'])

        check_productions_match(valid_actions['n'],
                                ['2013',
                                 '-1',
                                 '[<r,<f,n>>, r, f]',
                                 '[<r,<r,<f,n>>>, r, r, f]',
                                 '[<r,n>, r]'])

        check_productions_match(valid_actions['r'],
                                ['all_rows',
                                 '[<r,<m,<d,r>>>, r, m, d]',
                                 '[<r,<g,r>>, r, m]',
                                 '[<r,<g,r>>, r, f]',
                                 '[<r,<g,r>>, r, t]',
                                 '[<r,<c,r>>, r, m]',
                                 '[<r,<c,r>>, r, f]',
                                 '[<r,<f,<n,r>>>, r, f, n]',
                                 '[<r,<t,<s,r>>>, r, t, s]',
                                 '[<r,r>, r]'])

    def test_get_valid_actions_in_world_without_number_columns(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'first', 'title', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'corenlp_processed_tables' / 'TEST-6.table'
        table_context = TableQuestionContext.read_from_file(table_file, question_tokens)
        # The table does not have a number column.
        assert "number" not in table_context.column_types.values()
        world = WikiTablesVariableFreeWorld(table_context)
        actions = world.get_valid_actions()
        assert set(actions.keys()) == {
                "<r,<g,s>>",
                "<r,<c,r>>",
                "<r,<g,r>>",
                "<r,<t,<s,r>>>",
                "<n,<n,<n,d>>>",
                "<r,<m,<d,r>>>",
                "<r,r>",
                "<r,n>",
                "d",
                "n",
                "s",
                "m",
                "t",
                "r",
                "@start@",
                }
        assert set([str(type_) for type_ in world.get_basic_types()]) == {'n', 'd', 's', 'r', 't',
                                                                          'm', 'g', 'c'}
        check_productions_match(actions['s'],
                                ['[<r,<g,s>>, r, m]',
                                 '[<r,<g,s>>, r, t]'])

    def test_get_valid_actions_in_world_without_date_columns(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'first', 'title', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'corenlp_processed_tables' / 'TEST-4.table'
        table_context = TableQuestionContext.read_from_file(table_file, question_tokens)
        # The table does not have a date column.
        assert "date" not in table_context.column_types.values()
        world = WikiTablesVariableFreeWorld(table_context)
        actions = world.get_valid_actions()
        assert set(actions.keys()) == {
                "<r,<g,s>>",
                "<r,<f,<n,r>>>",
                "<r,<c,r>>",
                "<r,<g,r>>",
                "<r,<r,<f,n>>>",
                "<r,<t,<s,r>>>",
                "<n,<n,<n,d>>>",
                "<r,<f,n>>",
                "<r,r>",
                "<r,n>",
                "d",
                "n",
                "s",
                "t",
                "f",
                "r",
                "@start@",
                }
        assert set([str(type_) for type_ in world.get_basic_types()]) == {'n', 'd', 's', 'r', 't',
                                                                          'f', 'g', 'c'}
        check_productions_match(actions['s'],
                                ['[<r,<g,s>>, r, f]',
                                 '[<r,<g,s>>, r, t]'])

    def test_get_valid_actions_in_world_without_comparable_columns(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'first', 'title', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'corenlp_processed_tables' / 'TEST-1.table'
        table_context = TableQuestionContext.read_from_file(table_file, question_tokens)
        # The table does not have date or number columns.
        assert "date" not in table_context.column_types.values()
        assert "number" not in table_context.column_types.values()
        world = WikiTablesVariableFreeWorld(table_context)
        actions = world.get_valid_actions()
        assert set(actions.keys()) == {
                "<r,<g,s>>",
                "<r,<g,r>>",
                "<r,<t,<s,r>>>",
                "<n,<n,<n,d>>>",
                "<r,r>",
                "<r,n>",
                "d",
                "n",
                "s",
                "t",
                "r",
                "@start@",
                }
        assert set([str(type_) for type_ in world.get_basic_types()]) == {'n', 'd', 's', 'r', 't', 'g'}

    def test_parsing_logical_form_with_string_not_in_question_fails(self):
        logical_form_with_usl_a_league = """(select (filter_in all_rows string_column:league usl_a_league)
                                             date_column:year)"""
        logical_form_with_2013 = """(select (filter_date_greater all_rows date_column:year (date 2013 -1 -1))
                                     date_column:year)"""
        with self.assertRaises(ParsingError):
            self.world_with_2013.parse_logical_form(logical_form_with_usl_a_league)
            self.world_with_usl_a_league.parse_logical_form(logical_form_with_2013)

    @staticmethod
    def _get_alias(types_, name) -> str:
        if name in types_.generic_name_mapper.common_name_mapping:
            return types_.generic_name_mapper.get_alias(name)
        elif name in types_.string_column_name_mapper.common_name_mapping:
            return types_.string_column_name_mapper.get_alias(name)
        elif name in types_.number_column_name_mapper.common_name_mapping:
            return types_.number_column_name_mapper.get_alias(name)
        elif name in types_.date_column_name_mapper.common_name_mapping:
            return types_.date_column_name_mapper.get_alias(name)
        else:
            return types_.comparable_column_name_mapper.get_alias(name)

    def test_world_processes_logical_forms_correctly(self):
        logical_form = "(select (filter_in all_rows string_column:league string:usl_a_league) date_column:year)"
        expression = self.world_with_usl_a_league.parse_logical_form(logical_form)
        f = partial(self._get_alias, types)
        # Cells (and parts) get mapped to strings.
        # Column names are mapped in local name mapping. For the global names, we can get their
        # aliases from the name mapper.
        assert str(expression) == f"{f('select')}({f('filter_in')}({f('all_rows')},C2,string:usl_a_league),C0)"

    def test_world_gets_correct_actions(self):
        logical_form = "(select (filter_in all_rows string_column:league string:usl_a_league) date_column:year)"
        expression = self.world_with_usl_a_league.parse_logical_form(logical_form)
        expected_sequence = ['@start@ -> s', 's -> [<r,<g,s>>, r, m]', '<r,<g,s>> -> select',
                             'r -> [<r,<t,<s,r>>>, r, t, s]', '<r,<t,<s,r>>> -> filter_in',
                             'r -> all_rows', 't -> string_column:league', 's -> string:usl_a_league',
                             'm -> date_column:year']
        assert self.world_with_usl_a_league.get_action_sequence(expression) == expected_sequence

    def test_world_gets_logical_form_from_actions(self):
        logical_form = "(select (filter_in all_rows string_column:league string:usl_a_league) date_column:year)"
        expression = self.world_with_usl_a_league.parse_logical_form(logical_form)
        action_sequence = self.world_with_usl_a_league.get_action_sequence(expression)
        reconstructed_logical_form = self.world_with_usl_a_league.get_logical_form(action_sequence)
        assert logical_form == reconstructed_logical_form

    def test_world_processes_logical_forms_with_number_correctly(self):
        tokens = [Token(x) for x in ['when', 'was', 'the', 'attendance', 'higher', 'than', '3000',
                                     '?']]
        world = self._get_world_with_question_tokens(tokens)
        logical_form = """(select (filter_number_greater all_rows number_column:avg_attendance 3000)
                           date_column:year)"""
        expression = world.parse_logical_form(logical_form)
        f = partial(self._get_alias, types)
        # Cells (and parts) get mapped to strings.
        # Column names are mapped in local name mapping. For the global names, we can get their
        # aliases from the name mapper.
        assert str(expression) == f"{f('select')}({f('filter_number_greater')}({f('all_rows')},C6,num:3000),C0)"

    def test_world_processes_logical_forms_with_date_correctly(self):
        logical_form = """(select (filter_date_greater all_rows date_column:year (date 2013 -1 -1))
                           date_column:year)"""
        expression = self.world_with_2013.parse_logical_form(logical_form)
        f = partial(self._get_alias, types)
        # Cells (and parts) get mapped to strings.
        # Column names are mapped in local name mapping. For the global names, we can get their
        # aliases from the name mapper.
        assert str(expression) == \
        f"{f('select')}({f('filter_date_greater')}({f('all_rows')},C0,{f('date')}(num:2013,num:~1,num:~1)),C0)"

    def test_get_agenda(self):
        tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'n -> 2000',
                                           '<r,r> -> last',
                                           'm -> date_column:year'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'difference', 'in', 'attendance',
                                     'between', 'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # "year" column does not match because "years" occurs in the question.
        assert set(world.get_agenda()) == {'n -> 2001',
                                           'n -> 2005',
                                           '<r,<r,<f,n>>> -> diff'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'total', 'avg.', 'attendance', 'in',
                                     'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'n -> 2001',
                                           'n -> 2005',
                                           '<r,<f,n>> -> sum',
                                           'f -> number_column:avg_attendance'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,<c,r>> -> argmin', 'f -> number_column:avg_attendance'}
        tokens = [Token(x) for x in ['what', 'is', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,<f,n>> -> min', 'f -> number_column:avg_attendance'}
        tokens = [Token(x) for x in ['when', 'did', 'the', 'team', 'not', 'qualify', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'s -> string:qualify'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'at', 'least',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,<f,<n,r>>> -> filter_number_greater_equals',
                                           'f -> number_column:avg_attendance', 'n -> 7000'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'more', 'than',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,<f,<n,r>>> -> filter_number_greater',
                                           'f -> number_column:avg_attendance', 'n -> 7000'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'at', 'most',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,<f,<n,r>>> -> filter_number_lesser_equals',
                                           'f -> number_column:avg_attendance', 'n -> 7000'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'top', 'year', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,r> -> first', 'm -> date_column:year'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'year', 'in', 'the', 'bottom', 'row',
                                     '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<r,r> -> last', 'm -> date_column:year'}

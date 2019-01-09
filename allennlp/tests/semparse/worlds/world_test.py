# pylint: disable=no-self-use,invalid-name,protected-access
from overrides import overrides

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse import ParsingError, World
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import WikiTablesWorld


class FakeWorldWithoutRecursion(World):
    # pylint: disable=abstract-method
    @overrides
    def all_possible_actions(self):
        # The logical forms this grammar allows are
        # (unary_function argument)
        # (binary_function argument argument)
        actions = ['@start@ -> t',
                   't -> [<e,t>, e]',
                   '<e,t> -> unary_function',
                   '<e,t> -> [<e,<e,t>>, e]',
                   '<e,<e,t>> -> binary_function',
                   'e -> argument']
        return actions


class FakeWorldWithRecursion(FakeWorldWithoutRecursion):
    # pylint: disable=abstract-method
    @overrides
    def all_possible_actions(self):
        # In addition to the forms allowed by ``FakeWorldWithoutRecursion``, this world allows
        # (unary_function (identity .... (argument)))
        # (binary_function (identity .... (argument)) (identity .... (argument)))
        actions = super(FakeWorldWithRecursion, self).all_possible_actions()
        actions.extend(['e -> [<e,e>, e]',
                        '<e,e> -> identity'])
        return actions


class TestWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.world_without_recursion = FakeWorldWithoutRecursion()
        self.world_with_recursion = FakeWorldWithRecursion()

        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2004', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table.tsv'
        table_kg = TableQuestionKnowledgeGraph.read_from_file(table_file, question_tokens)
        self.wikitables_world = WikiTablesWorld(table_kg)

    def test_get_paths_to_root_without_recursion(self):
        argument_paths = self.world_without_recursion.get_paths_to_root('e -> argument')
        assert argument_paths == [['e -> argument', 't -> [<e,t>, e]', '@start@ -> t'],
                                  ['e -> argument', '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t']]
        unary_function_paths = self.world_without_recursion.get_paths_to_root('<e,t> -> unary_function')
        assert unary_function_paths == [['<e,t> -> unary_function', 't -> [<e,t>, e]',
                                         '@start@ -> t']]
        binary_function_paths = \
                self.world_without_recursion.get_paths_to_root('<e,<e,t>> -> binary_function')
        assert binary_function_paths == [['<e,<e,t>> -> binary_function',
                                          '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                          '@start@ -> t']]

    def test_get_paths_to_root_with_recursion(self):
        argument_paths = self.world_with_recursion.get_paths_to_root('e -> argument')
        # Argument now has 4 paths, and the two new paths are with the identity function occurring
        # (only once) within unary and binary functions.
        assert argument_paths == [['e -> argument', 't -> [<e,t>, e]', '@start@ -> t'],
                                  ['e -> argument', '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t'],
                                  ['e -> argument', 'e -> [<e,e>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t'],
                                  ['e -> argument', 'e -> [<e,e>, e]', '<e,t> -> [<e,<e,t>>, e]',
                                   't -> [<e,t>, e]', '@start@ -> t']]
        identity_paths = self.world_with_recursion.get_paths_to_root('<e,e> -> identity')
        # Two identity paths, one through each of unary and binary function productions.
        assert identity_paths == [['<e,e> -> identity', 'e -> [<e,e>, e]', 't -> [<e,t>, e]',
                                   '@start@ -> t'],
                                  ['<e,e> -> identity', 'e -> [<e,e>, e]',
                                   '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]', '@start@ -> t']]

    # The tests for get_action_sequence and get_logical_form need a concrete world to be useful;
    # we'll mostly use the NLVR world to test them, as it's a simpler world than the WikiTables
    # world.

    def test_get_action_sequence_removes_currying(self):
        world = self.wikitables_world
        logical_form = ("(argmax (number 1) (number 1) (fb:row.row.division fb:cell.2) "
                        "(reverse (lambda x ((reverse fb:row.row.index) (var x)))))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 'r -> [<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <n,r>]' in action_sequence

    def test_get_action_sequence_removes_and_retains_var_correctly(self):
        world = self.wikitables_world
        logical_form = ("((reverse fb:row.row.league) (argmin (number 1) (number 1) "
                        "(fb:type.object.type fb:type.row) "
                        "(reverse (lambda x ((reverse fb:row.row.index) (var x))))))")
        parsed_logical_form_without_var = world.parse_logical_form(logical_form)
        action_sequence_without_var = world.get_action_sequence(parsed_logical_form_without_var)
        assert '<#1,#1> -> var' not in action_sequence_without_var

        parsed_logical_form_with_var = world.parse_logical_form(logical_form,
                                                                remove_var_function=False)
        action_sequence_with_var = world.get_action_sequence(parsed_logical_form_with_var)
        assert '<#1,#1> -> var' in action_sequence_with_var

    def test_get_logical_form_handles_reverse(self):
        world = self.wikitables_world
        logical_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form

        logical_form = ("((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (argmax (number 1) "
                        "(number 1) (fb:row.row.league fb:cell.usl_a_league) (reverse (lambda x "
                        "((reverse fb:row.row.index) (var x)))))))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form

    def test_get_logical_form_handles_greater_than(self):
        world = self.wikitables_world
        action_sequence = ['@start@ -> c', 'c -> [<r,c>, r]', '<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
                           '<<#1,#2>,<#2,#1>> -> reverse', '<c,r> -> fb:row.row.league',
                           'r -> [<c,r>, c]', '<c,r> -> fb:row.row.year', 'c -> [<n,c>, n]',
                           '<n,c> -> fb:cell.cell.number', 'n -> [<nd,nd>, n]', '<nd,nd> -> >',
                           'n -> [<n,n>, n]', '<n,n> -> number', 'n -> 2004']
        logical_form = world.get_logical_form(action_sequence)
        expected_logical_form = ('((reverse fb:row.row.league) (fb:row.row.year '
                                 '(fb:cell.cell.number (> (number 2004)))))')
        assert logical_form == expected_logical_form

    def test_get_logical_form_handles_length_one_terminal_functions(self):
        world = self.wikitables_world
        logical_form = ("(- ((reverse fb:cell.cell.number) ((reverse fb:row.row.league) "
                        "(fb:row.row.year fb:cell.usl_a_league))) (number 1))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form

    def test_get_logical_form_adds_var_correctly(self):
        world = self.wikitables_world
        action_sequence = ['@start@ -> e', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                           '<<#1,#2>,<#2,#1>> -> reverse', '<e,r> -> fb:row.row.league',
                           'r -> [<d,<d,<#1,<<d,#1>,#1>>>>, d, d, r, <d,r>]',
                           '<d,<d,<#1,<<d,#1>,#1>>>> -> argmin', 'd -> [<e,d>, e]', '<e,d> -> number',
                           'e -> 1', 'd -> [<e,d>, e]', '<e,d> -> number', 'e -> 1',
                           'r -> [<#1,#1>, r]', '<#1,#1> -> fb:type.object.type', 'r -> fb:type.row',
                           '<d,r> -> [<<#1,#2>,<#2,#1>>, <r,d>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           "<r,d> -> ['lambda x', d]", 'd -> [<r,d>, r]',
                           '<r,d> -> [<<#1,#2>,<#2,#1>>, <d,r>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           '<d,r> -> fb:row.row.index', 'r -> x']
        logical_form = world.get_logical_form(action_sequence)
        assert '(var x)' in logical_form
        expected_logical_form = ("((reverse fb:row.row.league) (argmin (number 1) (number 1) "
                                 "(fb:type.object.type fb:type.row) "
                                 "(reverse (lambda x ((reverse fb:row.row.index) (var x))))))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        parsed_expected_logical_form = world.parse_logical_form(expected_logical_form)
        assert parsed_logical_form == parsed_expected_logical_form

    def test_get_logical_form_fails_with_unnecessary_add_var(self):
        world = self.wikitables_world
        action_sequence = ['@start@ -> e', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                           '<<#1,#2>,<#2,#1>> -> reverse', '<e,r> -> fb:row.row.league',
                           'r -> [<d,<d,<#1,<<d,#1>,#1>>>>, d, d, r, <d,r>]',
                           '<d,<d,<#1,<<d,#1>,#1>>>> -> argmin', 'd -> [<e,d>, e]', '<e,d> -> number',
                           'e -> 1', 'd -> [<e,d>, e]', '<e,d> -> number', 'e -> 1',
                           'r -> [<#1,#1>, r]', '<#1,#1> -> fb:type.object.type', 'r -> fb:type.row',
                           '<d,r> -> [<<#1,#2>,<#2,#1>>, <r,d>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           "<r,d> -> ['lambda x', d]", 'd -> [<r,d>, r]',
                           '<r,d> -> [<<#1,#2>,<#2,#1>>, <d,r>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           '<d,r> -> fb:row.row.index', 'r -> [<#1,#1>, r]', '<#1,#1> -> var', 'r -> x']
        with self.assertRaisesRegex(ParsingError, 'already had var'):
            world.get_logical_form(action_sequence)

# pylint: disable=no-self-use,invalid-name,protected-access
import json

from overrides import overrides

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse import World
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph
from allennlp.data.semparse.worlds import NlvrWorld, WikiTablesWorld
from allennlp.data.tokenizers import Token


class FakeWorldWithoutRecursion(World):
    # pylint: disable=abstract-method
    @overrides
    def all_possible_actions(self):
        # The logical forms this grammar allows are
        # (unary_function argument)
        # (binary_function argument argument)
        actions = ['@START@ -> t',
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


class WorldTest(AllenNlpTestCase):
    def setUp(self):
        super(WorldTest, self).setUp()
        self.world_without_recursion = FakeWorldWithoutRecursion()
        self.world_with_recursion = FakeWorldWithRecursion()

        test_filename = "tests/fixtures/data/nlvr/sample_data.jsonl"
        data = [json.loads(line)["structured_rep"] for line in open(test_filename).readlines()]
        self.nlvr_world = NlvrWorld(data[0])

        table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/sample_table.tsv")
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '?']]
        self.wikitables_world = WikiTablesWorld(table_kg, question_tokens)

    def test_get_paths_to_root_without_recursion(self):
        argument_paths = self.world_without_recursion.get_paths_to_root('e -> argument')
        assert argument_paths == [['e -> argument', 't -> [<e,t>, e]', '@START@ -> t'],
                                  ['e -> argument', '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                   '@START@ -> t']]
        unary_function_paths = self.world_without_recursion.get_paths_to_root('<e,t> -> unary_function')
        assert unary_function_paths == [['<e,t> -> unary_function', 't -> [<e,t>, e]',
                                         '@START@ -> t']]
        binary_function_paths = \
                self.world_without_recursion.get_paths_to_root('<e,<e,t>> -> binary_function')
        assert binary_function_paths == [['<e,<e,t>> -> binary_function',
                                          '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                          '@START@ -> t']]

    def test_get_paths_to_root_with_recursion(self):
        argument_paths = self.world_with_recursion.get_paths_to_root('e -> argument')
        # Argument now has 4 paths, and the two new paths are with the identity function occurring
        # (only once) within unary and binary functions.
        assert argument_paths == [['e -> argument', 't -> [<e,t>, e]', '@START@ -> t'],
                                  ['e -> argument', '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]',
                                   '@START@ -> t'],
                                  ['e -> argument', 'e -> [<e,e>, e]', 't -> [<e,t>, e]',
                                   '@START@ -> t'],
                                  ['e -> argument', 'e -> [<e,e>, e]', '<e,t> -> [<e,<e,t>>, e]',
                                   't -> [<e,t>, e]', '@START@ -> t']]
        identity_paths = self.world_with_recursion.get_paths_to_root('<e,e> -> identity')
        # Two identity paths, one through each of unary and binary function productions.
        assert identity_paths == [['<e,e> -> identity', 'e -> [<e,e>, e]', 't -> [<e,t>, e]',
                                   '@START@ -> t'],
                                  ['<e,e> -> identity', 'e -> [<e,e>, e]',
                                   '<e,t> -> [<e,<e,t>>, e]', 't -> [<e,t>, e]', '@START@ -> t']]

    # The tests for get_action_sequence and get_logical_form need a concrete world to be useful;
    # we'll mostly use the NLVR world to test them, as it's a simpler world than the WikiTables
    # world.

    def test_get_action_sequence_removes_and_retains_var_correctly(self):
        nlvr_world = self.nlvr_world
        logical_form = ("(assert_greater_equals \
                          (count (filter_equals all_boxes (lambda x (count (blue (object_in_box (var x))))) 0)) \
                          1)")
        parsed_logical_form_without_var = nlvr_world.parse_logical_form(logical_form)
        action_sequence_without_var = nlvr_world.get_action_sequence(parsed_logical_form_without_var)
        assert '<#1,#1> -> var' not in action_sequence_without_var

        parsed_logical_form_with_var = nlvr_world.parse_logical_form(logical_form,
                                                                     remove_var_function=False)
        action_sequence_with_var = nlvr_world.get_action_sequence(parsed_logical_form_with_var)
        assert '<#1,#1> -> var' in action_sequence_with_var

    def test_get_logical_form_with_real_logical_forms(self):
        nlvr_world = self.nlvr_world
        logical_form = ("(assert_greater_equals \
                          (count (filter_equals all_boxes (lambda x (count (blue (object_in_box (var x))))) 0)) \
                          1)")
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = nlvr_world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = nlvr_world.parse_logical_form(reconstructed_logical_form)
        # It makes more sense to compare parsed logical forms instead of actual logical forms
        # because there can be slight differences between the actual logical form strings, like
        # extra set of parentheses or spaces, which neither the type inference logic nor the
        # executor cares about. So the test shouldn't either.
        assert parsed_logical_form == parsed_reconstructed_logical_form
        assert nlvr_world.execute(logical_form) == nlvr_world.execute(reconstructed_logical_form)

        logical_form = "(assert_equals (color (circle (touch_wall (all_objects)))) color_black)"
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = nlvr_world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = nlvr_world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form
        assert nlvr_world.execute(logical_form) == nlvr_world.execute(reconstructed_logical_form)

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

    def test_get_logical_form_handles_length_one_terminal_functions(self):
        world = self.wikitables_world
        logical_form = ("(- ((reverse fb:cell.cell.number) ((reverse fb:row.row.league) "
                        "(fb:row.row.year fb:cell.usl_a_league))) (number 1))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form


    def test_get_logical_form_with_decoded_action_sequence(self):
        # This is identical to the previous test except that we are testing it on a real action
        # sequence the decoder produced.
        nlvr_world = self.nlvr_world
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> [<#1,<#1,t>>, b]',
                           '<#1,<#1,t>> -> assert_not_equals', 'b -> all_boxes', 'b -> all_boxes']
        logical_form = nlvr_world.get_logical_form(action_sequence)
        # Note that while the grammar allows this logical form, it is not valid according to the
        # executor because `assert_*` functions take attributes (counts, shapes and colors) as both
        # the arguments. The signature is "<#1,<#1,t>>" because we do not have a generic attribute
        # type, and hence it is more lenient than it should be. We should ideally have hierarchical
        # types to deal with these.
        assert logical_form == '(assert_not_equals all_boxes all_boxes)'

    def test_get_logical_form_fails_with_incomplete_action_sequence(self):
        nlvr_world = self.nlvr_world
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> [<#1,<#1,t>>, b]',
                           '<#1,<#1,t>> -> assert_not_equals']
        with self.assertRaises(AssertionError):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_fails_with_action_sequence_in_wrong_order(self):
        nlvr_world = self.nlvr_world
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> [<#1,<#1,t>>, b]',
                           'b -> all_boxes', '<#1,<#1,t>> -> assert_not_equals', 'b -> all_boxes']
        with self.assertRaises(AssertionError):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_adds_var_correctly(self):
        nlvr_world = self.nlvr_world
        action_sequence = ['@START@ -> t', 't -> [<e,t>, e]', '<e,t> -> [<e,<e,t>>, e]',
                           '<e,<e,t>> -> assert_greater_equals', 'e -> [<#1,e>, b]',
                           '<#1,e> -> count', 'b -> [<e,b>, e]', '<e,b> -> [<<b,e>,<e,b>>, <b,e>]',
                           '<<b,e>,<e,b>> -> [<b,<<b,#1>,<#1,b>>>, b]',
                           '<b,<<b,#1>,<#1,b>>> -> filter_equals', 'b -> all_boxes',
                           "<b,e> -> ['lambda x', e]", 'e -> [<#1,e>, o]', '<#1,e> -> count',
                           'o -> [<o,o>, o]', '<o,o> -> blue', 'o -> [<b,o>, b]',
                           '<b,o> -> object_in_box', 'b -> x', 'e -> 0', 'e -> 1']
        logical_form = nlvr_world.get_logical_form(action_sequence)
        expected_logical_form = ("(assert_greater_equals \
                                 (count (filter_equals all_boxes (lambda x (count (blue \
                                 (object_in_box (var (x)))))) 0)) 1)")
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        parsed_expected_logical_form = nlvr_world.parse_logical_form(expected_logical_form)
        assert parsed_logical_form == parsed_expected_logical_form

    def test_get_logical_form_fails_with_unnecessary_add_var(self):
        nlvr_world = self.nlvr_world
        action_sequence = ['@START@ -> t', 't -> [<e,t>, e]', '<e,t> -> [<e,<e,t>>, e]',
                           '<e,<e,t>> -> assert_greater_equals', 'e -> [<#1,e>, b]',
                           '<#1,e> -> count', 'b -> [<e,b>, e]', '<e,b> -> [<<b,e>,<e,b>>, <b,e>]',
                           '<<b,e>,<e,b>> -> [<b,<<b,#1>,<#1,b>>>, b]',
                           '<b,<<b,#1>,<#1,b>>> -> filter_equals', 'b -> all_boxes',
                           "<b,e> -> ['lambda x', e]", 'e -> [<#1,e>, o]', '<#1,e> -> count',
                           'o -> [<o,o>, o]', '<o,o> -> blue', 'o -> [<b,o>, b]',
                           '<b,o> -> object_in_box', 'b -> [<#1,#1>, b]', '<#1,#1> -> var',
                           'b -> x', 'e -> 0', 'e -> 1']
        with self.assertRaises(RuntimeError):
            nlvr_world.get_logical_form(action_sequence)

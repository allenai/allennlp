# pylint: disable=no-self-use,invalid-name
import json

from allennlp.data.semparse.worlds.nlvr_world import NlvrWorld
from allennlp.common.testing import AllenNlpTestCase


class TestNlvrWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = "tests/fixtures/data/nlvr/sample_data.jsonl"
        data = [json.loads(line)["structured_rep"] for line in open(test_filename).readlines()]
        self.worlds = [NlvrWorld(rep) for rep in data]

    def test_logical_form_with_assert_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There is a circle closely touching a corner of a box." and label is "True".
        logical_form_true = "(assert_greater_equals (count (touch_corner (circle (all_objects)))) 1)"
        # Should evaluate to True.
        assert nlvr_world.execute(logical_form_true)
        logical_form_false = "(assert_equals (count (touch_corner (circle (all_objects)))) 0)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form_false)

    def test_logical_form_with_filter_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There is a box without a blue item." and label is "False".
        logical_form = "(assert_greater_equals \
                         (count (filter_equals all_boxes (lambda x (count (blue (object_in_box (var x))))) 0)) \
                         1)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_same_color_executes_correctly(self):
        nlvr_world = self.worlds[1]
        # Utterance is "There is exactly one tower with two blocks of the same color." and label is "True".
        logical_form = "(assert_equals \
                         (count \
                          (filter_equals all_boxes (lambda x (count (same_color (object_in_box (var x))))) 2)) \
                         1)"
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_same_shape_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There are less than three black objects of the same shape" and label is "False".
        logical_form = "(assert_lesser (count (same_shape (black (all_objects)))) 3)"
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_touch_wall_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There are two black circles touching a wall" and label is "False".
        logical_form = "(assert_greater_equals (count (touch_wall (black (circle (all_objects))))) 2)"
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_not_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There are at most two medium triangles not touching a wall." and label is "True".
        logical_form = ("(assert_lesser_equals (count (negate_filter (touch_wall) \
                                                                     (medium (triangle (all_objects))))) 2)")
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_color_comparison_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "The color of the circle touching the wall is black." and label is "True".
        logical_form = "(assert_equals (color (circle (touch_wall (all_objects)))) color_black)"
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_object_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = "(assert_equals (color (circle (touch_wall (all_objects)))) color_black)"
        expression = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['@START@ -> t', 't -> [<c,t>, c]', '<c,t> -> [<#1,<#1,t>>, c]',
                                   '<#1,<#1,t>> -> assert_equals', 'c -> [<o,c>, o]', '<o,c> -> color',
                                   'o -> [<o,o>, o]', '<o,o> -> circle', 'o -> [<o,o>, o]',
                                   '<o,o> -> touch_wall', 'o -> all_objects', 'c -> color_black']

    def test_logical_form_with_box_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = ("(assert_greater_equals \
                          (count (filter_equals all_boxes (lambda x (count (blue (object_in_box (var (x)))))) 0)) \
                          1)")
        expression = nlvr_world.parse_logical_form(logical_form, remove_var_function=False)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['@START@ -> t', 't -> [<e,t>, e]', '<e,t> -> [<e,<e,t>>, e]',
                                   '<e,<e,t>> -> assert_greater_equals', 'e -> [<#1,e>, b]', '<#1,e> -> count',
                                   'b -> [<e,b>, e]', '<e,b> -> [<<b,e>,<e,b>>, <b,e>]',
                                   '<<b,e>,<e,b>> -> [<b,<<b,#1>,<#1,b>>>, b]',
                                   '<b,<<b,#1>,<#1,b>>> -> filter_equals', 'b -> all_boxes',
                                   "<b,e> -> ['lambda x', e]", 'e -> [<#1,e>, o]', '<#1,e> -> count',
                                   'o -> [<o,o>, o]', '<o,o> -> blue', 'o -> [<b,o>, b]', '<b,o> -> object_in_box',
                                   'b -> [<#1,#1>, b]', '<#1,#1> -> var', 'b -> x', 'e -> 0', 'e -> 1']

    def test_get_logical_form_with_real_logical_forms(self):
        nlvr_world = self.worlds[0]
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

    def test_get_logical_form_with_decoded_action_sequence(self):
        # This is identical to the previous test except that we are testing it on a real action
        # sequence the decoder produced.
        nlvr_world = self.worlds[0]
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
        nlvr_world = self.worlds[0]
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> [<#1,<#1,t>>, b]',
                           '<#1,<#1,t>> -> assert_not_equals']
        with self.assertRaises(AssertionError):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_fails_with_action_sequence_in_wrong_order(self):
        nlvr_world = self.worlds[0]
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> [<#1,<#1,t>>, b]',
                           'b -> all_boxes', '<#1,<#1,t>> -> assert_not_equals', 'b -> all_boxes']
        with self.assertRaises(AssertionError):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_action_sequence_removes_and_retains_var_correctly(self):
        nlvr_world = self.worlds[0]
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

    def test_get_logical_form_adds_var_correctly(self):
        nlvr_world = self.worlds[0]
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
        nlvr_world = self.worlds[0]
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

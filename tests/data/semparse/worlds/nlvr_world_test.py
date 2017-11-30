# pylint: disable=no-self-use,invalid-name
import json

from allennlp.data.semparse.worlds.nlvr_world import NlvrWorld
from allennlp.common.testing import AllenNlpTestCase


class TestNlvrWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = "tests/fixtures/data/nlvr/sample_data.json"
        data = [json.loads(line)["structured_rep"] for line in open(test_filename).readlines()]
        self.worlds = [NlvrWorld(rep) for rep in data]

    def test_logical_form_with_assert_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There is a circle closely touching a corner of a box." and label is "True".
        logical_form_true = "(assert_greater_equals (count (touch_corner (circle (all_objects)))) 1)"
        # Should evaluate to True.
        assert nlvr_world.execute(logical_form_true)
        logical_form_false = "(assert_equals (count (touch_corner (circle(all_objects)))) 0)"
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
        assert action_sequence == ['t', 't -> [<c,t>, c]', '<c,t> -> [<#1,<#1,t>>, c]',
                                   '<#1,<#1,t>> -> assert_equals', 'c -> [<o,c>, o]', '<o,c> -> color',
                                   'o -> [<o,o>, o]', '<o,o> -> circle', 'o -> [<o,o>, o]',
                                   '<o,o> -> touch_wall', 'o -> all_objects', 'c -> color_black']

    def test_logical_form_with_box_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = ("(assert_greater_equals \
                          (count (filter_equals all_boxes (lambda x (count (blue (object_in_box (var (x)))))) 0)) \
                          1)")
        expression = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['t', 't -> [<e,t>, e]', '<e,t> -> [<#1,<#1,t>>, e]',
                                   '<#1,<#1,t>> -> assert_greater_equals', 'e -> [<#1,e>, b]', '<#1,e> -> count',
                                   'b -> [<e,b>, e]', '<e,b> -> [<<b,e>,<e,b>>, <b,e>]',
                                   '<<b,e>,<e,b>> -> [<b,<<b,#1>,<#1,b>>>, b]',
                                   '<b,<<b,#1>,<#1,b>>> -> filter_equals', 'b -> all_boxes',
                                   "<b,e> -> ['lambda x', e]", 'e -> [<#1,e>, o]', '<#1,e> -> count',
                                   'o -> [<o,o>, o]', '<o,o> -> blue', 'o -> [<b,o>, b]', '<b,o> -> object_in_box',
                                   'b -> [<#1,#1>, b]', '<#1,#1> -> var', 'b -> x', 'e -> 0', 'e -> 1']

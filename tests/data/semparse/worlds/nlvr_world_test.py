# pylint: disable=no-self-use,invalid-name
import json

from allennlp.data.semparse.worlds.world import ParsingError
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
        logical_form_true = "(object_count_greater_equals (touch_corner (circle (all_objects))) 1)"
        # Should evaluate to True.
        assert nlvr_world.execute(logical_form_true)
        logical_form_false = "(object_count_equals (touch_corner (circle (all_objects))) 0)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form_false)

    def test_logical_form_with_box_filter_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There is a box without a blue item." and label is "False".
        logical_form = "(box_exists (member_color_none_equals all_boxes color_blue))"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_box_filter_within_object_filter_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There are atleast three blue items in boxes with blue items" and label
        # is "True".
        logical_form = "(object_count_greater_equals \
                            (object_in_box (member_color_any_equals all_boxes color_blue)) 3)"
        # Should evaluate to True.
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_same_color_executes_correctly(self):
        nlvr_world = self.worlds[1]
        # Utterance is "There are exactly two blocks of the same color." and label is "True".
        logical_form = "(object_count_equals (same_color all_objects) 2)"
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_same_shape_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There are less than three black objects of the same shape" and label is "False".
        logical_form = "(object_count_lesser (same_shape (black (all_objects))) 3)"
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_touch_wall_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There are two black circles touching a wall" and label is "False".
        logical_form = "(object_count_greater_equals (touch_wall (black (circle (all_objects)))) 2)"
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_not_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There are at most two medium triangles not touching a wall." and label is "True".
        logical_form = ("(object_count_lesser_equals (negate_filter (touch_wall) \
                                                                     (medium (triangle (all_objects)))) 2)")
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_color_comparison_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "The color of the circle touching the wall is black." and label is "True".
        logical_form = "(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_object_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = "(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        expression = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['@START@ -> t', 't -> [<c,t>, c]', '<c,t> -> [<o,<c,t>>, o]',
                                   '<o,<c,t>> -> object_color_all_equals', 'o -> [<o,o>, o]',
                                   '<o,o> -> circle', 'o -> [<o,o>, o]', '<o,o> -> touch_wall',
                                   'o -> all_objects', 'c -> color_black']

    def test_logical_form_with_box_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = "(box_exists (member_color_none_equals all_boxes color_blue))"
        expression = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> box_exists',
                                   'b -> [<c,b>, c]', '<c,b> -> [<b,<c,b>>, b]',
                                   '<b,<c,b>> -> member_color_none_equals', 'b -> all_boxes',
                                   'c -> color_blue']

    def test_get_logical_form_with_real_logical_forms(self):
        nlvr_world = self.worlds[0]
        logical_form = ("(box_count_greater_equals (member_color_count_equals all_boxes 0) 1)")
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = nlvr_world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = nlvr_world.parse_logical_form(reconstructed_logical_form)
        # It makes more sense to compare parsed logical forms instead of actual logical forms.
        assert parsed_logical_form == parsed_reconstructed_logical_form
        assert nlvr_world.execute(logical_form) == nlvr_world.execute(reconstructed_logical_form)
        logical_form = "(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = nlvr_world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = nlvr_world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form
        assert nlvr_world.execute(logical_form) == nlvr_world.execute(reconstructed_logical_form)

    def test_get_logical_form_fails_with_incomplete_action_sequence(self):
        nlvr_world = self.worlds[0]
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> box_exists']
        with self.assertRaises(ParsingError):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_fails_with_action_sequence_in_wrong_order(self):
        nlvr_world = self.worlds[0]
        action_sequence = ['@START@ -> t', 't -> [<b,t>, b]', '<b,t> -> box_exists',
                           'b -> [<c,b>, c]', '<c,b> -> [<b,<c,b>>, b]',
                           'b -> all_boxes', '<b,<c,b>> -> member_color_none_equals',
                           'c -> color_blue']
        with self.assertRaises(ParsingError):
            nlvr_world.get_logical_form(action_sequence)

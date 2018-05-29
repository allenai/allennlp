# pylint: disable=no-self-use,invalid-name
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.worlds.nlvr_world import NlvrWorld


class TestNlvrWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = self.FIXTURES_ROOT / "data" / "nlvr" / "sample_ungrouped_data.jsonl"
        data = [json.loads(line)["structured_rep"] for line in open(test_filename).readlines()]
        self.worlds = [NlvrWorld(rep) for rep in data]
        # y_loc increases as we go down from top to bottom, and x_loc from left to right. That is,
        # the origin is at the top-left corner.
        custom_rep = [[{"y_loc": 79, "size": 20, "type": "triangle", "x_loc": 27, "color": "Yellow"},
                       {"y_loc": 55, "size": 10, "type": "circle", "x_loc": 47, "color": "Black"}],
                      [{"y_loc": 44, "size": 30, "type": "square", "x_loc": 10, "color": "#0099ff"},
                       {"y_loc": 74, "size": 30, "type": "square", "x_loc": 40, "color": "Yellow"}],
                      [{"y_loc": 60, "size": 10, "type": "triangle", "x_loc": 12, "color": "#0099ff"}]]
        self.custom_world = NlvrWorld(custom_rep)

    def test_get_action_sequence_removes_currying_for_all_nlvr_functions(self):
        world = self.worlds[0]
        # box_color_filter
        logical_form = "(member_color_all_equals all_boxes color_blue)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 'b -> [<b,<c,b>>, b, c]' in action_sequence

        # box_shape_filter
        logical_form = "(member_shape_all_equals all_boxes shape_square)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 'b -> [<b,<s,b>>, b, s]' in action_sequence

        # box_count_filter
        logical_form = "(member_count_equals all_boxes 3)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 'b -> [<b,<e,b>>, b, e]' in action_sequence

        # assert_color
        logical_form = "(object_color_all_equals all_objects color_blue)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 't -> [<o,<c,t>>, o, c]' in action_sequence

        # assert_shape
        logical_form = "(object_shape_all_equals all_objects shape_square)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 't -> [<o,<s,t>>, o, s]' in action_sequence

        # assert_box_count
        logical_form = "(box_count_equals all_boxes 1)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 't -> [<b,<e,t>>, b, e]' in action_sequence

        # assert_object_count
        logical_form = "(object_count_equals all_objects 1)"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert 't -> [<o,<e,t>>, o, e]' in action_sequence

    def test_logical_form_with_assert_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There is a circle closely touching a corner of a box." and label is "True".
        logical_form_true = "(object_count_greater_equals (touch_corner (circle (all_objects))) 1)"
        assert nlvr_world.execute(logical_form_true) is True
        logical_form_false = "(object_count_equals (touch_corner (circle (all_objects))) 0)"
        assert nlvr_world.execute(logical_form_false) is False

    def test_logical_form_with_box_filter_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There is a box without a blue item." and label is "False".
        logical_form = "(box_exists (member_color_none_equals all_boxes color_blue))"
        assert nlvr_world.execute(logical_form) is False

    def test_logical_form_with_box_filter_within_object_filter_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There are at least three blue items in boxes with blue items" and label
        # is "True".
        logical_form = "(object_count_greater_equals \
                            (object_in_box (member_color_any_equals all_boxes color_blue)) 3)"
        assert nlvr_world.execute(logical_form) is True

    def test_logical_form_with_same_color_executes_correctly(self):
        nlvr_world = self.worlds[1]
        # Utterance is "There are exactly two blocks of the same color." and label is "True".
        logical_form = "(object_count_equals (same_color all_objects) 2)"
        assert nlvr_world.execute(logical_form) is True

    def test_logical_form_with_same_shape_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There are less than three black objects of the same shape" and label is "False".
        logical_form = "(object_count_lesser (same_shape (black (all_objects))) 3)"
        assert nlvr_world.execute(logical_form) is False

    def test_logical_form_with_touch_wall_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "There are two black circles touching a wall" and label is "False".
        logical_form = "(object_count_greater_equals (touch_wall (black (circle (all_objects)))) 2)"
        assert nlvr_world.execute(logical_form) is False

    def test_logical_form_with_not_executes_correctly(self):
        nlvr_world = self.worlds[2]
        # Utterance is "There are at most two medium triangles not touching a wall." and label is "True".
        logical_form = ("(object_count_lesser_equals ((negate_filter touch_wall) \
                                                                     (medium (triangle (all_objects)))) 2)")
        assert nlvr_world.execute(logical_form) is True

    def test_logical_form_with_color_comparison_executes_correctly(self):
        nlvr_world = self.worlds[0]
        # Utterance is "The color of the circle touching the wall is black." and label is "True".
        logical_form = "(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        assert nlvr_world.execute(logical_form) is True

    def test_logical_form_with_object_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = "(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        expression = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['@start@ -> t', 't -> [<o,<c,t>>, o, c]',
                                   '<o,<c,t>> -> object_color_all_equals', 'o -> [<o,o>, o]',
                                   '<o,o> -> circle', 'o -> [<o,o>, o]', '<o,o> -> touch_wall',
                                   'o -> all_objects', 'c -> color_black']

    def test_logical_form_with_box_filter_returns_correct_action_sequence(self):
        nlvr_world = self.worlds[0]
        logical_form = "(box_exists (member_color_none_equals all_boxes color_blue))"
        expression = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(expression)
        assert action_sequence == ['@start@ -> t', 't -> [<b,t>, b]', '<b,t> -> box_exists',
                                   'b -> [<b,<c,b>>, b, c]', '<b,<c,b>> -> member_color_none_equals',
                                   'b -> all_boxes', 'c -> color_blue']

    def test_spatial_relations_return_objects_in_the_same_box(self):
        # "above", "below", "top", "bottom" are relations defined only for objects within the same
        # box. So they should not return objects from other boxes.
        world = self.custom_world
        # Asserting that the color of the objects above the yellow triangle is only black (it is not
        # yellow or blue, which are colors of objects from other boxes)
        assert world.execute("(object_color_all_equals (above (yellow (triangle all_objects)))"
                             " color_black)") is True
        # Asserting that the only shape below the blue square is a square.
        assert world.execute("(object_shape_all_equals (below (blue (square all_objects)))"
                             " shape_square)") is True
        # Asserting the shape of the object at the bottom in the box with a circle is triangle.
        logical_form = ("(object_shape_all_equals (bottom (object_in_box"
                        " (member_shape_any_equals all_boxes shape_circle))) shape_triangle)")
        assert world.execute(logical_form) is True

        # Asserting the shape of the object at the top of the box with all squares is a square (!).
        logical_form = ("(object_shape_all_equals (top (object_in_box"
                        " (member_shape_all_equals all_boxes shape_square))) shape_square)")
        assert world.execute(logical_form) is True

    def test_touch_object_executes_correctly(self):
        world = self.custom_world
        # Assert that there is a yellow square touching a blue square.
        assert world.execute("(object_exists (yellow (square (touch_object (blue "
                             "(square all_objects))))))") is True
        # Assert that the triangle does not touch the circle (they are out of vertical range).
        assert world.execute("(object_shape_none_equals (touch_object (triangle all_objects))"
                             " shape_circle)") is True

    def test_spatial_relations_with_objects_from_different_boxes(self):
        # When the objects are from different boxes, top and bottom should return objects from
        # respective boxes.
        world = self.custom_world
        # There are triangles in two boxes, so top should return the top objects from both boxes.
        assert world.execute("(object_count_equals (top (object_in_box (member_shape_any_equals "
                             "all_boxes shape_triangle))) 2)") is True

    def test_count_with_all_equals_throws_execution_error(self):
        # "*_all_equals" is a comparison valid only for sets (of colors and shapes). A comparison
        # with count should use "*_equals" instead.
        world = self.custom_world
        with self.assertRaises(ExecutionError):
            world.execute("(object_count_all_equals (top (object_in_box (member_shape_any_equals "
                          "all_boxes shape_triangle))) 2)")

    def test_shape_with_equals_throws_execution_error(self):
        # "*_all_equals" is a comparison valid only for counts. A comparison with sets should use
        # "*_equals", "*_any_equals" or "*_non_equals" instead.
        world = self.custom_world
        with self.assertRaises(ExecutionError):
            world.execute("(object_shape_equals (top (object_in_box (member_shape_any_equals "
                          "all_boxes shape_triangle))) shape_triangle)")

    def test_same_and_different_execute_correctly(self):
        world = self.custom_world
        # All the objects in the box with two objects of the same shape are squares.
        assert world.execute("(object_shape_all_equals "
                             "(object_in_box (member_shape_same (member_count_equals all_boxes 2)))"
                             " shape_square)") is True
        # There is a circle in the box with objects of different shapes.
        assert world.execute("(object_shape_any_equals (object_in_box "
                             "(member_shape_different all_boxes)) shape_circle)") is True

    def test_get_agenda_for_sentence(self):
        world = self.worlds[0]
        agenda = world.get_agenda_for_sentence("there is a tower with exactly two yellow blocks")
        assert set(agenda) == set(['c -> color_yellow', '<b,t> -> box_exists', 'e -> 2'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "the bottom of a box.")
        assert set(agenda) == set(['<o,o> -> yellow', '<o,o> -> touch_bottom', 'e -> 1'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "the right wall of a box.")
        assert set(agenda) == set(['<o,o> -> yellow', '<o,o> -> touch_right', 'e -> 1'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "the left wall of a box.")
        assert set(agenda) == set(['<o,o> -> yellow', '<o,o> -> touch_left', 'e -> 1'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "a wall of a box.")
        assert set(agenda) == set(['<o,o> -> yellow', '<o,o> -> touch_wall', 'e -> 1'])
        agenda = world.get_agenda_for_sentence("There is exactly one square touching any edge")
        assert set(agenda) == set(['<o,o> -> square', '<o,o> -> touch_wall', 'e -> 1'])
        agenda = world.get_agenda_for_sentence("There is exactly one square not touching any edge")
        assert set(agenda) == set(['<o,o> -> square', '<o,o> -> touch_wall', 'e -> 1',
                                   '<<o,o>,<o,o>> -> negate_filter'])
        agenda = world.get_agenda_for_sentence("There is only 1 tower with 1 blue block at the base")
        assert set(agenda) == set(['<o,o> -> blue', 'e -> 1', '<o,o> -> bottom', 'e -> 1'])
        agenda = world.get_agenda_for_sentence("There is only 1 tower that has 1 blue block at the top")
        assert set(agenda) == set(['<o,o> -> blue', 'e -> 1', '<o,o> -> top', 'e -> 1',
                                   'b -> all_boxes'])
        agenda = world.get_agenda_for_sentence("There is exactly one square touching the blue "
                                               "triangle")
        assert set(agenda) == set(['<o,o> -> square', '<o,o> -> blue', '<o,o> -> triangle',
                                   '<o,o> -> touch_object', 'e -> 1'])

    def test_get_agenda_for_sentence_correctly_adds_object_filters(self):
        # In logical forms that contain "box_exists" at the top, there can never be object filtering
        # operations like "blue", "square" etc. In those cases, strings like "blue" and "square" in
        # sentences should map to "color_blue" and "shape_square" respectively.
        world = self.worlds[0]
        agenda = world.get_agenda_for_sentence("there is a box with exactly two yellow triangles "
                                               "touching the top edge")
        assert "<o,o> -> yellow" not in agenda
        assert "c -> color_yellow" in agenda
        assert "<o,o> -> triangle" not in agenda
        assert "s -> shape_triangle" in agenda
        assert "<o,o> -> touch_top" not in agenda
        agenda = world.get_agenda_for_sentence("there are exactly two yellow triangles touching the"
                                               " top edge")
        assert "<o,o> -> yellow" in agenda
        assert "c -> color_yellow" not in agenda
        assert "<o,o> -> triangle" in agenda
        assert "s -> shape_triangle" not in agenda
        assert "<o,o> -> touch_top" in agenda

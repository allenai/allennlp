# pylint: disable=no-self-use,invalid-name
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages import NlvrLanguage
from allennlp.semparse.domain_languages.nlvr_language import Box


class TestNlvrLanguage(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = self.FIXTURES_ROOT / "data" / "nlvr" / "sample_ungrouped_data.jsonl"
        data = [json.loads(line)["structured_rep"] for line in open(test_filename).readlines()]
        box_lists = [[Box(object_reps, i) for i, object_reps in enumerate(box_rep)] for box_rep in data]
        self.languages = [NlvrLanguage(boxes) for boxes in box_lists]
        # y_loc increases as we go down from top to bottom, and x_loc from left to right. That is,
        # the origin is at the top-left corner.
        custom_rep = [[{"y_loc": 79, "size": 20, "type": "triangle", "x_loc": 27, "color": "Yellow"},
                       {"y_loc": 55, "size": 10, "type": "circle", "x_loc": 47, "color": "Black"}],
                      [{"y_loc": 44, "size": 30, "type": "square", "x_loc": 10, "color": "#0099ff"},
                       {"y_loc": 74, "size": 30, "type": "square", "x_loc": 40, "color": "Yellow"}],
                      [{"y_loc": 60, "size": 10, "type": "triangle", "x_loc": 12, "color": "#0099ff"}]]
        self.custom_language = NlvrLanguage([Box(object_rep, i) for i, object_rep in enumerate(custom_rep)])

    def test_logical_form_with_assert_executes_correctly(self):
        executor = self.languages[0]
        # Utterance is "There is a circle closely touching a corner of a box." and label is "True".
        logical_form_true = "(object_count_greater_equals (touch_corner (circle (all_objects))) 1)"
        assert executor.execute(logical_form_true) is True
        logical_form_false = "(object_count_equals (touch_corner (circle (all_objects))) 9)"
        assert executor.execute(logical_form_false) is False

    def test_logical_form_with_box_filter_executes_correctly(self):
        executor = self.languages[2]
        # Utterance is "There is a box without a blue item." and label is "False".
        logical_form = "(box_exists (member_color_none_equals all_boxes color_blue))"
        assert executor.execute(logical_form) is False

    def test_logical_form_with_box_filter_within_object_filter_executes_correctly(self):
        executor = self.languages[2]
        # Utterance is "There are at least three blue items in boxes with blue items" and label
        # is "True".
        logical_form = "(object_count_greater_equals \
                            (object_in_box (member_color_any_equals all_boxes color_blue)) 3)"
        assert executor.execute(logical_form) is True

    def test_logical_form_with_same_color_executes_correctly(self):
        executor = self.languages[1]
        # Utterance is "There are exactly two blocks of the same color." and label is "True".
        logical_form = "(object_count_equals (same_color all_objects) 2)"
        assert executor.execute(logical_form) is True

    def test_logical_form_with_same_shape_executes_correctly(self):
        executor = self.languages[0]
        # Utterance is "There are less than three black objects of the same shape" and label is "False".
        logical_form = "(object_count_lesser (same_shape (black (all_objects))) 3)"
        assert executor.execute(logical_form) is False

    def test_logical_form_with_touch_wall_executes_correctly(self):
        executor = self.languages[0]
        # Utterance is "There are two black circles touching a wall" and label is "False".
        logical_form = "(object_count_greater_equals (touch_wall (black (circle (all_objects)))) 2)"
        assert executor.execute(logical_form) is False

    def test_logical_form_with_not_executes_correctly(self):
        executor = self.languages[2]
        # Utterance is "There are at most two medium triangles not touching a wall." and label is "True".
        logical_form = ("(object_count_lesser_equals ((negate_filter touch_wall) "
                        "(medium (triangle (all_objects)))) 2)")
        assert executor.execute(logical_form) is True

    def test_logical_form_with_color_comparison_executes_correctly(self):
        executor = self.languages[0]
        # Utterance is "The color of the circle touching the wall is black." and label is "True".
        logical_form = "(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        assert executor.execute(logical_form) is True

    def test_spatial_relations_return_objects_in_the_same_box(self):
        # "above", "below", "top", "bottom" are relations defined only for objects within the same
        # box. So they should not return objects from other boxes.
        # Asserting that the color of the objects above the yellow triangle is only black (it is not
        # yellow or blue, which are colors of objects from other boxes)
        assert self.custom_language.execute("(object_color_all_equals (above (yellow (triangle all_objects)))"
                                            " color_black)") is True
        # Asserting that the only shape below the blue square is a square.
        assert self.custom_language.execute("(object_shape_all_equals (below (blue (square all_objects)))"
                                            " shape_square)") is True
        # Asserting the shape of the object at the bottom in the box with a circle is triangle.
        logical_form = ("(object_shape_all_equals (bottom (object_in_box"
                        " (member_shape_any_equals all_boxes shape_circle))) shape_triangle)")
        assert self.custom_language.execute(logical_form) is True

        # Asserting the shape of the object at the top of the box with all squares is a square (!).
        logical_form = ("(object_shape_all_equals (top (object_in_box"
                        " (member_shape_all_equals all_boxes shape_square))) shape_square)")
        assert self.custom_language.execute(logical_form) is True

    def test_touch_object_executes_correctly(self):
        # Assert that there is a yellow square touching a blue square.
        assert self.custom_language.execute("(object_exists (yellow (square (touch_object (blue "
                                            "(square all_objects))))))") is True
        # Assert that the triangle does not touch the circle (they are out of vertical range).
        assert self.custom_language.execute("(object_shape_none_equals (touch_object (triangle all_objects))"
                                            " shape_circle)") is True

    def test_spatial_relations_with_objects_from_different_boxes(self):
        # When the objects are from different boxes, top and bottom should return objects from
        # respective boxes.
        # There are triangles in two boxes, so top should return the top objects from both boxes.
        assert self.custom_language.execute("(object_count_equals (top (object_in_box (member_shape_any_equals "
                                            "all_boxes shape_triangle))) 2)") is True

    def test_same_and_different_execute_correctly(self):
        # All the objects in the box with two objects of the same shape are squares.
        assert self.custom_language.execute("(object_shape_all_equals "
                                            "(object_in_box (member_shape_same (member_count_equals all_boxes 2)))"
                                            " shape_square)") is True
        # There is a circle in the box with objects of different shapes.
        assert self.custom_language.execute("(object_shape_any_equals (object_in_box "
                                            "(member_shape_different all_boxes)) shape_circle)") is True

    def test_get_action_sequence_handles_multi_arg_functions(self):
        language = self.languages[0]
        # box_color_filter
        logical_form = "(box_exists (member_color_all_equals all_boxes color_blue))"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'Set[Box] -> [<Set[Box],Color:Set[Box]>, Set[Box], Color]' in action_sequence

        # box_shape_filter
        logical_form = "(box_exists (member_shape_all_equals all_boxes shape_square))"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'Set[Box] -> [<Set[Box],Shape:Set[Box]>, Set[Box], Shape]' in action_sequence

        # box_count_filter
        logical_form = "(box_exists (member_count_equals all_boxes 3))"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'Set[Box] -> [<Set[Box],int:Set[Box]>, Set[Box], int]' in action_sequence

        # assert_color
        logical_form = "(object_color_all_equals all_objects color_blue)"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'bool -> [<Set[Object],Color:bool>, Set[Object], Color]' in action_sequence

        # assert_shape
        logical_form = "(object_shape_all_equals all_objects shape_square)"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'bool -> [<Set[Object],Shape:bool>, Set[Object], Shape]' in action_sequence

        # assert_box_count
        logical_form = "(box_count_equals all_boxes 1)"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'bool -> [<Set[Box],int:bool>, Set[Box], int]' in action_sequence

        # assert_object_count
        logical_form = "(object_count_equals all_objects 1)"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert 'bool -> [<Set[Object],int:bool>, Set[Object], int]' in action_sequence

    def test_logical_form_with_object_filter_returns_correct_action_sequence(self):
        language = self.languages[0]
        logical_form = "(object_color_all_equals (circle (touch_wall all_objects)) color_black)"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> bool',
                                   'bool -> [<Set[Object],Color:bool>, Set[Object], Color]',
                                   '<Set[Object],Color:bool> -> object_color_all_equals',
                                   'Set[Object] -> [<Set[Object]:Set[Object]>, Set[Object]]',
                                   '<Set[Object]:Set[Object]> -> circle',
                                   'Set[Object] -> [<Set[Object]:Set[Object]>, Set[Object]]',
                                   '<Set[Object]:Set[Object]> -> touch_wall',
                                   'Set[Object] -> all_objects',
                                   'Color -> color_black']

    def test_logical_form_with_negate_filter_returns_correct_action_sequence(self):
        language = self.languages[0]
        logical_form = "(object_exists ((negate_filter touch_wall) all_objects))"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        negate_filter_production = ('<Set[Object]:Set[Object]> -> '
                                    '[<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>>, '
                                    '<Set[Object]:Set[Object]>]')
        assert action_sequence == ['@start@ -> bool',
                                   'bool -> [<Set[Object]:bool>, Set[Object]]',
                                   '<Set[Object]:bool> -> object_exists',
                                   'Set[Object] -> [<Set[Object]:Set[Object]>, Set[Object]]',
                                   negate_filter_production,
                                   '<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>> -> negate_filter',
                                   '<Set[Object]:Set[Object]> -> touch_wall',
                                   'Set[Object] -> all_objects']

    def test_logical_form_with_box_filter_returns_correct_action_sequence(self):
        language = self.languages[0]
        logical_form = "(box_exists (member_color_none_equals all_boxes color_blue))"
        action_sequence = language.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> bool',
                                   'bool -> [<Set[Box]:bool>, Set[Box]]',
                                   '<Set[Box]:bool> -> box_exists',
                                   'Set[Box] -> [<Set[Box],Color:Set[Box]>, Set[Box], Color]',
                                   '<Set[Box],Color:Set[Box]> -> member_color_none_equals',
                                   'Set[Box] -> all_boxes',
                                   'Color -> color_blue']

    def test_get_agenda_for_sentence(self):
        language = self.languages[0]
        agenda = language.get_agenda_for_sentence("there is a tower with exactly two yellow blocks")
        assert set(agenda) == set(['Color -> color_yellow', '<Set[Box]:bool> -> box_exists', 'int -> 2'])
        agenda = language.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                                  "the bottom of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_bottom', 'int -> 1'])
        agenda = language.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                                  "the right wall of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_right', 'int -> 1'])
        agenda = language.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                                  "the left wall of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_left', 'int -> 1'])
        agenda = language.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                                  "a wall of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_wall', 'int -> 1'])
        agenda = language.get_agenda_for_sentence("There is exactly one square touching any edge")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> square',
                                   '<Set[Object]:Set[Object]> -> touch_wall', 'int -> 1'])
        agenda = language.get_agenda_for_sentence("There is exactly one square not touching any edge")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> square',
                                   '<Set[Object]:Set[Object]> -> touch_wall', 'int -> 1',
                                   '<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>> -> negate_filter'])
        agenda = language.get_agenda_for_sentence("There is only 1 tower with 1 blue block at the base")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> blue', 'int -> 1',
                                   '<Set[Object]:Set[Object]> -> bottom', 'int -> 1'])
        agenda = language.get_agenda_for_sentence("There is only 1 tower that has 1 blue block at the top")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> blue', 'int -> 1',
                                   '<Set[Object]:Set[Object]> -> top', 'int -> 1',
                                   'Set[Box] -> all_boxes'])
        agenda = language.get_agenda_for_sentence("There is exactly one square touching the blue "
                                                  "triangle")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> square',
                                   '<Set[Object]:Set[Object]> -> blue', '<Set[Object]:Set[Object]> -> triangle',
                                   '<Set[Object]:Set[Object]> -> touch_object', 'int -> 1'])

    def test_get_agenda_for_sentence_correctly_adds_object_filters(self):
        # In logical forms that contain "box_exists" at the top, there can never be object filtering
        # operations like "blue", "square" etc. In those cases, strings like "blue" and "square" in
        # sentences should map to "color_blue" and "shape_square" respectively.
        language = self.languages[0]
        agenda = language.get_agenda_for_sentence("there is a box with exactly two yellow triangles "
                                                  "touching the top edge")
        assert "<Set[Object]:Set[Object]> -> yellow" not in agenda
        assert "Color -> color_yellow" in agenda
        assert "<Set[Object]:Set[Object]> -> triangle" not in agenda
        assert "Shape -> shape_triangle" in agenda
        assert "<Set[Object]:Set[Object]> -> touch_top" not in agenda
        agenda = language.get_agenda_for_sentence("there are exactly two yellow triangles touching the"
                                                  " top edge")
        assert "<Set[Object]:Set[Object]> -> yellow" in agenda
        assert "Color -> color_yellow" not in agenda
        assert "<Set[Object]:Set[Object]> -> triangle" in agenda
        assert "Shape -> shape_triangle" not in agenda
        assert "<Set[Object]:Set[Object]> -> touch_top" in agenda

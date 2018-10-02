# pylint: disable=no-self-use,invalid-name
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.nlvr_world import NlvrWorld


class TestNlvrWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = self.FIXTURES_ROOT / "data" / "nlvr" / "sample_ungrouped_data.jsonl"
        data = [json.loads(line)["structured_rep"] for line in open(test_filename).readlines()]
        self.worlds = [NlvrWorld(rep) for rep in data]

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

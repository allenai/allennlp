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

    def test_get_agenda_for_sentence(self):
        world = self.worlds[0]
        agenda = world.get_agenda_for_sentence("there is a tower with exactly two yellow blocks")
        assert set(agenda) == set(['Color -> color_yellow', '<Set[Box]:bool> -> box_exists', 'int -> 2'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "the bottom of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_bottom', 'int -> 1'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "the right wall of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_right', 'int -> 1'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "the left wall of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_left', 'int -> 1'])
        agenda = world.get_agenda_for_sentence("There is at most one yellow item closely touching "
                                               "a wall of a box.")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> yellow',
                                   '<Set[Object]:Set[Object]> -> touch_wall', 'int -> 1'])
        agenda = world.get_agenda_for_sentence("There is exactly one square touching any edge")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> square',
                                   '<Set[Object]:Set[Object]> -> touch_wall', 'int -> 1'])
        agenda = world.get_agenda_for_sentence("There is exactly one square not touching any edge")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> square',
                                   '<Set[Object]:Set[Object]> -> touch_wall', 'int -> 1',
                                   '<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>> -> negate_filter'])
        agenda = world.get_agenda_for_sentence("There is only 1 tower with 1 blue block at the base")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> blue', 'int -> 1',
                                   '<Set[Object]:Set[Object]> -> bottom', 'int -> 1'])
        agenda = world.get_agenda_for_sentence("There is only 1 tower that has 1 blue block at the top")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> blue', 'int -> 1',
                                   '<Set[Object]:Set[Object]> -> top', 'int -> 1',
                                   'Set[Box] -> all_boxes'])
        agenda = world.get_agenda_for_sentence("There is exactly one square touching the blue "
                                               "triangle")
        assert set(agenda) == set(['<Set[Object]:Set[Object]> -> square',
                                   '<Set[Object]:Set[Object]> -> blue', '<Set[Object]:Set[Object]> -> triangle',
                                   '<Set[Object]:Set[Object]> -> touch_object', 'int -> 1'])

    def test_get_agenda_for_sentence_correctly_adds_object_filters(self):
        # In logical forms that contain "box_exists" at the top, there can never be object filtering
        # operations like "blue", "square" etc. In those cases, strings like "blue" and "square" in
        # sentences should map to "color_blue" and "shape_square" respectively.
        world = self.worlds[0]
        agenda = world.get_agenda_for_sentence("there is a box with exactly two yellow triangles "
                                               "touching the top edge")
        assert "<Set[Object]:Set[Object]> -> yellow" not in agenda
        assert "Color -> color_yellow" in agenda
        assert "<Set[Object]:Set[Object]> -> triangle" not in agenda
        assert "Shape -> shape_triangle" in agenda
        assert "<Set[Object]:Set[Object]> -> touch_top" not in agenda
        agenda = world.get_agenda_for_sentence("there are exactly two yellow triangles touching the"
                                               " top edge")
        assert "<Set[Object]:Set[Object]> -> yellow" in agenda
        assert "Color -> color_yellow" not in agenda
        assert "<Set[Object]:Set[Object]> -> triangle" in agenda
        assert "Shape -> shape_triangle" not in agenda
        assert "<Set[Object]:Set[Object]> -> touch_top" in agenda

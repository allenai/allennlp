# pylint: disable=no-self-use,invalid-name
import json

from allennlp.data.semparse.worlds.nlvr_world import NLVRWorld
from allennlp.common.testing import AllenNlpTestCase


class TestNLVRWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = "tests/fixtures/data/nlvr/sample_data.json"
        self.data = [json.loads(line) for line in open(test_filename).readlines()]

    def test_logical_form_with_assert_executes_correctly(self):
        nlvr_world = NLVRWorld(self.data[0]["structured_rep"])
        # Utterance is "There is a circle closely touching a corner of a box." and label is "True".
        logical_form_true = "assert_greater_equals(count(touch_corner(circle(all_objects))) 1)"
        # Should evaluate to True.
        assert nlvr_world.execute(logical_form_true)
        logical_form_false = "assert_equals(count(touch_corner(circle(all_objects))) 0)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form_false)

    def test_logical_form_with_filter_executes_correctly(self):
        nlvr_world = NLVRWorld(self.data[2]["structured_rep"])
        # Utterance is "There is a box without a blue item." and label is "False".
        logical_form = "assert_greater_equals(count(filter_equals(all_boxes count(blue(object_in_box)) 0)) 1)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_same_color_executes_correctly(self):
        nlvr_world = NLVRWorld(self.data[1]["structured_rep"])
        # Utterance is "There is exactly one tower with two blocks of the same color." and label is "True".
        logical_form = "assert_equals(count(filter_equals(all_boxes count(same_color(object_in_box)) 2)) 1)"
        assert nlvr_world.execute(logical_form)

    def test_logical_form_with_same_shape_executes_correctly(self):
        nlvr_world = NLVRWorld(self.data[0]["structured_rep"])
        # Utterance is "There are less than three black objects of the same shape" and label is "False".
        logical_form = "assert_lesser(count(same_shape(black(all_objects))) 3)"
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_touch_wall_executes_correctly(self):
        nlvr_world = NLVRWorld(self.data[0]["structured_rep"])
        # Utterance is "There are two black circles touching a wall" and label is "False".
        logical_form = "assert_greater_equals(count(touch_wall(black(circle(all_objects)))) 2)"
        assert not nlvr_world.execute(logical_form)

    def test_logical_form_with_not_executes_correctly(self):
        nlvr_world = NLVRWorld(self.data[2]["structured_rep"])
        # Utterance is "There are at most two medium triangles not touching a wall." and label is "True".
        logical_form = "assert_lesser_equals(count(negate_filter(touch_wall medium(triangle(all_objects)))) 2)"
        assert nlvr_world.execute(logical_form)

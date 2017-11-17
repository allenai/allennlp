# pylint: disable=no-self-use,invalid-name
import json

from allennlp.data.semparse.worlds.nlvr_world import NLVRWorld
from allennlp.common.testing import AllenNlpTestCase


class TestNLVRWorldRepresentation(AllenNlpTestCase):
    def test_logical_form_with_assert_executes_correctly(self):
        test_filename = "tests/fixtures/data/nlvr/sample_data.json"
        data = json.loads(open(test_filename).readlines()[0])
        nlvr_world = NLVRWorld(data["structured_rep"])
        # Utterance is "There is a circle closely touching a corner of a box." and label is "True".
        logical_form_true = "assert_greater_equals(count(touch_corner(circle(all_objects))) 1)"
        # Should evaluate to True.
        assert nlvr_world.execute(logical_form_true)
        logical_form_false = "assert_equals(count(touch_corner(circle(all_objects))) 0)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form_false)

    def test_logical_form_with_filter_executes_correctly(self):
        test_filename = "tests/fixtures/data/nlvr/sample_data.json"
        data = json.loads(open(test_filename).readlines()[2])
        nlvr_world = NLVRWorld(data["structured_rep"])
        # Utterance is "There is a box without a blue item." and label is "False".
        logical_form = "assert_greater_equals(count(filter_equals(all_boxes count(blue(object_in_box)) 1)) 1)"
        # Should evaluate to False.
        assert not nlvr_world.execute(logical_form)

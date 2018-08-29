# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import QuACReader
from allennlp.common.testing import AllenNlpTestCase


class TestQuACReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        params = Params({'lazy': lazy, 'num_context_answers': 2,})
        reader = QuACReader.from_params(params)
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'quac_sample.json'))
        instances = ensure_list(instances)

        assert instances[0].fields["question"].sequence_length() == 6
        assert instances[0].fields["yesno_list"].sequence_length() == 6
        assert [t.text for t in instances[0].fields["question"].field_list[0].tokens[:3]] == ["What", "was", "the"]

        assert len(instances) == 2
        passage_length = len(instances[0].fields["passage"].tokens)

        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == ["DJ", "Kool", "Herc"]
        assert [x.label for x in instances[0].fields["yesno_list"].field_list] == ["x", "x", "y", "x", "x", "x"]
        assert [x.label for x in instances[0].fields["followup_list"].field_list] == ["y", "m", "m", "n", "m", "y"]
        assert instances[0].fields["p1_answer_marker"].field_list[0].labels == ["O"] * passage_length

        # Check the previous answer marking here
        prev_1_list = ["O"] * passage_length
        prev_2_list = ["O"] * passage_length
        q0_span_start = instances[0].fields['span_start'].field_list[0].sequence_index
        q0_span_end = instances[0].fields['span_end'].field_list[0].sequence_index
        prev_1_list[q0_span_start] = "<{0:d}_{1:s}>".format(1, "start")
        prev_1_list[q0_span_end] = "<{0:d}_{1:s}>".format(1, "end")
        prev_2_list[q0_span_start] = "<{0:d}_{1:s}>".format(2, "start")
        prev_2_list[q0_span_end] = "<{0:d}_{1:s}>".format(2, "end")
        for passage_index in range(q0_span_start + 1, q0_span_end):
            prev_1_list[passage_index] = "<{0:d}_{1:s}>".format(1, "in")
            prev_2_list[passage_index] = "<{0:d}_{1:s}>".format(2, "in")

        assert instances[0].fields["p1_answer_marker"].field_list[1].labels == prev_1_list
        assert instances[0].fields["p2_answer_marker"].field_list[2].labels == prev_2_list

# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import DQAReader
from allennlp.common.testing import AllenNlpTestCase

class TestDQAReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        params = Params({
                'lazy': lazy
                })
        reader = DQAReader.from_params(params)
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'dqa_sample.json'))
        instances = ensure_list(instances)
        assert len(instances) == 3

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["Which", "American", "-"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == ["The", "Nobel", "Prize"]
        assert instances[0].fields["span_start"].sequence_index == 12
        assert instances[0].fields["span_end"].sequence_index == 13

        assert [t.text for t in instances[1].fields["question"].tokens[:3]] == ["Which", "American", "-"]
        assert [t.text for t in instances[1].fields["passage"].tokens[:3]] == ["Why", "Do", "n’t"]
        assert [t.text for t in instances[1].fields["passage"].tokens[-3:]] == ["adults", ",", "and"]
        assert instances[1].fields["span_start"].sequence_index == 38
        assert instances[1].fields["span_end"].sequence_index == 39

        assert [t.text for t in instances[2].fields["question"].tokens[:3]] == ["Where", "in", "England"]
        assert [t.text for t in instances[2].fields["passage"].tokens[:3]] == ["Judi", "Dench", "-"]
        assert [t.text for t in instances[2].fields["passage"].tokens[-3:]] == [")", "(", "special"]
        assert instances[2].fields["span_start"].sequence_index == 16
        assert instances[2].fields["span_end"].sequence_index == 16

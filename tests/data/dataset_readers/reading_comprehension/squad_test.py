# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import SquadReader

class TestSquadReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SquadReader(lazy=lazy)
        instances = ensure_list(reader.read('tests/fixtures/data/squad.json'))
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["To", "whom", "did"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == ["Architecturally", ",", "the"]
        assert [t.text for t in instances[0].fields["passage"].tokens[-3:]] == ["of", "Mary", "."]
        assert instances[0].fields["span_start"].sequence_index == 102
        assert instances[0].fields["span_end"].sequence_index == 104

        assert [t.text for t in instances[1].fields["question"].tokens[:3]] == ["What", "sits", "on"]
        assert [t.text for t in instances[1].fields["passage"].tokens[:3]] == ["Architecturally", ",", "the"]
        assert [t.text for t in instances[1].fields["passage"].tokens[-3:]] == ["of", "Mary", "."]
        assert instances[1].fields["span_start"].sequence_index == 17
        assert instances[1].fields["span_end"].sequence_index == 23

        # We're checking this case because I changed the answer text to only have a partial
        # annotation for the last token, which happens occasionally in the training data.  We're
        # making sure we get a reasonable output in that case here.
        assert ([t.text for t in instances[3].fields["question"].tokens[:3]] ==
                ["Which", "individual", "worked"])
        assert [t.text for t in instances[3].fields["passage"].tokens[:3]] == ["In", "1882", ","]
        assert [t.text for t in instances[3].fields["passage"].tokens[-3:]] == ["Nuclear", "Astrophysics", "."]
        span_start = instances[3].fields["span_start"].sequence_index
        span_end = instances[3].fields["span_end"].sequence_index
        answer_tokens = instances[3].fields["passage"].tokens[span_start:(span_end + 1)]
        expected_answer_tokens = ["Father", "Julius", "Nieuwland"]
        assert [t.text for t in answer_tokens] == expected_answer_tokens

    def test_can_build_from_params(self):
        reader = SquadReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'

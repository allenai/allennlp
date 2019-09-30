import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import SquadReader
from allennlp.common.testing import AllenNlpTestCase


class TestSquadReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SquadReader(lazy=lazy)
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "squad.json"))
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["To", "whom", "did"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[0].fields["passage"].tokens[-3:]] == ["of", "Mary", "."]
        assert instances[0].fields["span_start"].sequence_index == 102
        assert instances[0].fields["span_end"].sequence_index == 104

        assert [t.text for t in instances[1].fields["question"].tokens[:3]] == [
            "What",
            "sits",
            "on",
        ]
        assert [t.text for t in instances[1].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[1].fields["passage"].tokens[-3:]] == ["of", "Mary", "."]
        assert instances[1].fields["span_start"].sequence_index == 17
        assert instances[1].fields["span_end"].sequence_index == 23

        # We're checking this case because I changed the answer text to only have a partial
        # annotation for the last token, which happens occasionally in the training data.  We're
        # making sure we get a reasonable output in that case here.
        assert [t.text for t in instances[3].fields["question"].tokens[:3]] == [
            "Which",
            "individual",
            "worked",
        ]
        assert [t.text for t in instances[3].fields["passage"].tokens[:3]] == ["In", "1882", ","]
        assert [t.text for t in instances[3].fields["passage"].tokens[-3:]] == [
            "Nuclear",
            "Astrophysics",
            ".",
        ]
        span_start = instances[3].fields["span_start"].sequence_index
        span_end = instances[3].fields["span_end"].sequence_index
        answer_tokens = instances[3].fields["passage"].tokens[span_start : (span_end + 1)]
        expected_answer_tokens = ["Father", "Julius", "Nieuwland"]
        assert [t.text for t in answer_tokens] == expected_answer_tokens

    def test_can_build_from_params(self):
        reader = SquadReader.from_params(Params({}))

        assert reader._tokenizer.__class__.__name__ == "WordTokenizer"
        assert reader._token_indexers["tokens"].__class__.__name__ == "SingleIdTokenIndexer"

    def test_length_limit_works(self):
        # We're making sure the length of the text is correct if length limit is provided.
        reader = SquadReader(
            passage_length_limit=30, question_length_limit=10, skip_invalid_examples=True
        )
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "squad.json"))
        assert len(instances[0].fields["question"].tokens) == 10
        assert len(instances[0].fields["passage"].tokens) == 30
        # invalid examples where all the answers exceed the passage length should be skipped.
        assert len(instances) == 3

        # Length limit still works if we do not skip the invalid examples
        reader = SquadReader(
            passage_length_limit=30, question_length_limit=10, skip_invalid_examples=False
        )
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "squad.json"))
        assert len(instances[0].fields["question"].tokens) == 10
        assert len(instances[0].fields["passage"].tokens) == 30
        # invalid examples should not be skipped.
        assert len(instances) == 5

        # Make sure the answer texts does not change, so that the evaluation will not be affected
        reader_unlimited = SquadReader(
            passage_length_limit=30, question_length_limit=10, skip_invalid_examples=False
        )
        instances_unlimited = ensure_list(
            reader_unlimited.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "squad.json")
        )
        for instance_x, instance_y in zip(instances, instances_unlimited):
            print(instance_x.fields["metadata"]["answer_texts"])
            assert set(instance_x.fields["metadata"]["answer_texts"]) == set(
                instance_y.fields["metadata"]["answer_texts"]
            )

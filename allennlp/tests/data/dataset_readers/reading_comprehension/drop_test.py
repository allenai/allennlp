# pylint: disable=no-self-use,invalid-name, protected-access
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import DropReader
from allennlp.common.testing import AllenNlpTestCase


class TestDropReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = DropReader(lazy=lazy)
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'drop.json'))
        assert len(instances) == 19

        print(instances[0])
        instance = instances[0]
        assert [t.text for t in instance["question"][:3]] == ["What", "happened", "second"]
        assert [t.text for t in instance["passage"][:3]] == ["The", "Port", "of"]
        assert [t.text for t in instance["passage"][-3:]] == ["cruise", "ships", "."]

        # Note that the last number in here is added as padding in case we don't find any numbers
        # in a particular passage.
        assert [f.sequence_index for f in instance["number_indices"]] == [
                16, 30, 36, 41, 52, 64, 80, 89, 147, 153, 166, 174, 177, 206, 245, 252, 267, 279,
                283, 288, 296, -1
                ]
        assert [t.text for t in instance["numbers_in_passage"]] == [
                "1", "25", "2014", "5", "2018", "1", "2", "1", "54", "52", "6", "60", "58", "2010",
                "67", "2010", "1996", "3", "1", "6", "1", "0"]
        assert len(instance["answer_as_passage_spans"]) == 1
        assert instance["answer_as_passage_spans"][0] == (46, 47)
        assert len(instance["answer_as_question_spans"]) == 1
        assert instance["answer_as_question_spans"][0] == (5, 6)
        assert len(instance["answer_as_add_sub_expressions"]) == 1
        assert instance["answer_as_add_sub_expressions"][0].labels == [0,] * 22
        assert len(instance["answer_as_counts"]) == 1
        assert instance["answer_as_counts"][0].label == -1
        assert instance['metadata'].metadata.keys() == {'original_passage',
                                                        'passage_token_offsets',
                                                        'question_token_offsets',
                                                        'question_tokens', 'passage_tokens',
                                                        'number_tokens', 'number_indices',
                                                        'answer_texts', 'original_question',
                                                        'original_numbers', 'passage_id',
                                                        'question_id', 'answer_info',
                                                        'answer_annotations'}

    def test_can_build_from_params(self):
        reader = DropReader.from_params(Params({}))
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'

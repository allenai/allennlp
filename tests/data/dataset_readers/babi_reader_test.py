import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import BabiReader
from allennlp.common.testing import AllenNlpTestCase


class TestBAbIReader:
    @pytest.mark.parametrize(
        "keep_sentences, lazy", [(False, False), (False, True), (True, False), (True, True)]
    )
    def test_read_from_file(self, keep_sentences, lazy):
        reader = BabiReader(keep_sentences=keep_sentences, lazy=lazy)
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "babi.txt"))
        assert len(instances) == 8

        if keep_sentences:
            assert [t.text for t in instances[0].fields["context"][3].tokens[3:]] == [
                "of",
                "wolves",
                ".",
            ]
            assert [t.sequence_index for t in instances[0].fields["supports"]] == [0, 1]
        else:
            assert [t.text for t in instances[0].fields["context"].tokens[7:9]] == ["afraid", "of"]

    def test_can_build_from_params(self):
        reader = BabiReader.from_params(Params({"keep_sentences": True}))

        assert reader._keep_sentences
        assert reader._token_indexers["tokens"].__class__.__name__ == "SingleIdTokenIndexer"

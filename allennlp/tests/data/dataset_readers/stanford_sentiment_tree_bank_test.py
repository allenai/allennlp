import pytest

from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestStanfordSentimentTreebankReader:
    sst_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "sst.txt"

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = StanfordSentimentTreeBankDatasetReader(lazy=lazy)
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["The", "actors", "are", "fantastic", "."], "label": "4"}
        instance2 = {"tokens": ["It", "was", "terrible", "."], "label": "0"}
        instance3 = {"tokens": ["Chomp", "chomp", "!"], "label": "2"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    def test_use_subtrees(self):
        reader = StanfordSentimentTreeBankDatasetReader(use_subtrees=True)
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["The", "actors", "are", "fantastic", "."], "label": "4"}
        instance2 = {"tokens": ["The", "actors"], "label": "2"}
        instance3 = {"tokens": ["The"], "label": "2"}

        assert len(instances) == 21
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    def test_3_class(self):
        reader = StanfordSentimentTreeBankDatasetReader(granularity="3-class")
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["The", "actors", "are", "fantastic", "."], "label": "2"}
        instance2 = {"tokens": ["It", "was", "terrible", "."], "label": "0"}
        instance3 = {"tokens": ["Chomp", "chomp", "!"], "label": "1"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    def test_2_class(self):
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class")
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["The", "actors", "are", "fantastic", "."], "label": "1"}
        instance2 = {"tokens": ["It", "was", "terrible", "."], "label": "0"}

        assert len(instances) == 2
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]

    def test_from_params(self):

        params = Params({"use_subtrees": True, "granularity": "5-class"})
        reader = StanfordSentimentTreeBankDatasetReader.from_params(params)
        assert reader._use_subtrees is True
        assert reader._granularity == "5-class"

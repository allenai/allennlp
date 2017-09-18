# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import SnliReader
from allennlp.common.testing import AllenNlpTestCase


class TestSnliReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SnliReader()
        dataset = reader.read('tests/fixtures/data/snli.jsonl')

        instance1 = {"premise": ["A", "person", "on", "a", "horse", "jumps", "over", "a", "broken",
                                 "down", "airplane", "."],
                     "hypothesis": ["A", "person", "is", "training", "his", "horse", "for", "a",
                                    "competition", "."],
                     "label": "neutral"}

        instance2 = {"premise": ["A", "person", "on", "a", "horse", "jumps", "over", "a", "broken",
                                 "down", "airplane", "."],
                     "hypothesis": ["A", "person", "is", "at", "a", "diner", ",", "ordering", "an",
                                    "omelette", "."],
                     "label": "contradiction"}
        instance3 = {"premise": ["A", "person", "on", "a", "horse", "jumps", "over", "a", "broken",
                                 "down", "airplane", "."],
                     "hypothesis": ["A", "person", "is", "outdoors", ",", "on", "a", "horse", "."],
                     "label": "entailment"}

        assert len(dataset.instances) == 3
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["premise"].tokens] == instance1["premise"]
        assert [t.text for t in fields["hypothesis"].tokens] == instance1["hypothesis"]
        assert fields["label"].label == instance1["label"]
        fields = dataset.instances[1].fields
        assert [t.text for t in fields["premise"].tokens] == instance2["premise"]
        assert [t.text for t in fields["hypothesis"].tokens] == instance2["hypothesis"]
        assert fields["label"].label == instance2["label"]
        fields = dataset.instances[2].fields
        assert [t.text for t in fields["premise"].tokens] == instance3["premise"]
        assert [t.text for t in fields["hypothesis"].tokens] == instance3["hypothesis"]
        assert fields["label"].label == instance3["label"]

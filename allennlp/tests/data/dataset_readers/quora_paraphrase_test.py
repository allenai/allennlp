import pytest

from allennlp.data.dataset_readers import QuoraParaphraseDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestQuoraParaphraseReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = QuoraParaphraseDatasetReader(lazy=lazy)
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "quora_paraphrase.tsv")
        instances = ensure_list(instances)

        instance1 = {
            "premise": "What should I do to avoid sleeping in class ?".split(),
            "hypothesis": "How do I not sleep in a boring class ?".split(),
            "label": "1",
        }

        instance2 = {
            "premise": "Do women support each other more than men do ?".split(),
            "hypothesis": "Do women need more compliments than men ?".split(),
            "label": "0",
        }

        instance3 = {
            "premise": "How can one root android devices ?".split(),
            "hypothesis": "How do I root an Android device ?".split(),
            "label": "1",
        }

        assert len(instances) == 3

        for instance, expected_instance in zip(instances, [instance1, instance2, instance3]):
            fields = instance.fields
            assert [t.text for t in fields["premise"].tokens] == expected_instance["premise"]
            assert [t.text for t in fields["hypothesis"].tokens] == expected_instance["hypothesis"]
            assert fields["label"].label == expected_instance["label"]

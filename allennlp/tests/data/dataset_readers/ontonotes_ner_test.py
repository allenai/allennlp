import pytest

from allennlp.data.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestOntonotesNamedEntityRecognitionReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = OntonotesNamedEntityRecognition(lazy=lazy)
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / "conll_2012" / "subdomain")
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == [
            "Mali",
            "government",
            "officials",
            "say",
            "the",
            "woman",
            "'s",
            "confession",
            "was",
            "forced",
            ".",
        ]
        assert fields["tags"].labels == ["B-GPE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == [
            "The",
            "prosecution",
            "rested",
            "its",
            "case",
            "last",
            "month",
            "after",
            "four",
            "months",
            "of",
            "hearings",
            ".",
        ]
        assert fields["tags"].labels == [
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-DATE",
            "I-DATE",
            "O",
            "B-DATE",
            "I-DATE",
            "O",
            "O",
            "O",
        ]

        fields = instances[2].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["tags"].labels == [
            "B-PERSON",
            "I-PERSON",
            "B-WORK_OF_ART",
            "I-WORK_OF_ART",
            "O",
        ]

    def test_ner_reader_can_filter_by_domain(self):
        conll_reader = OntonotesNamedEntityRecognition(domain_identifier="subdomain2")
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / "conll_2012")
        instances = ensure_list(instances)
        assert len(instances) == 1

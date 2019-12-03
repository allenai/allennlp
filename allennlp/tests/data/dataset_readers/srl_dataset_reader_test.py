import pytest

from allennlp.data.dataset_readers.semantic_role_labeling import (
    SrlReader,
    _convert_tags_to_wordpiece_tags,
)
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestSrlReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = SrlReader(lazy=lazy)
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
        assert fields["verb_indicator"].labels[3] == 1
        assert fields["tags"].labels == [
            "B-ARG0",
            "I-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "O",
        ]
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[3]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        fields = instances[1].fields
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
        assert fields["verb_indicator"].labels[8] == 1
        assert fields["tags"].labels == [
            "O",
            "O",
            "O",
            "O",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "B-V",
            "B-ARG2",
            "O",
        ]
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[8]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        fields = instances[2].fields
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
        assert fields["verb_indicator"].labels[2] == 1
        assert fields["tags"].labels == [
            "B-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "O",
        ]
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[2]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        fields = instances[3].fields
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
        assert fields["verb_indicator"].labels[11] == 1
        assert fields["tags"].labels == [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-V",
            "O",
        ]
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[11]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["verb_indicator"].labels == [0, 0, 0, 0, 0]
        assert fields["tags"].labels == ["O", "O", "O", "O", "O"]
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] is None
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

    def test_srl_reader_can_filter_by_domain(self):

        conll_reader = SrlReader(domain_identifier="subdomain2")
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / "conll_2012")
        instances = ensure_list(instances)
        # If we'd included the folder, we'd have 9 instances.
        assert len(instances) == 2


class TestBertSrlReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.reader = SrlReader(bert_model_name="bert-base-uncased")

    def test_convert_tags_to_wordpiece_tags(self):

        offsets = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        original = [
            "B-ARG0",
            "I-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "O",
        ]
        wordpiece_tags = [
            "O",
            "B-ARG0",
            "I-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "O",
            "O",
        ]
        converted = _convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == wordpiece_tags

        offsets = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
        converted = _convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == [
            "O",
            "B-ARG0",
            "I-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "O",
        ]

        offsets = [1, 3, 5]
        original = ["B-ARG", "B-V", "O"]
        converted = _convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == ["O", "B-ARG", "B-V", "I-V", "O", "O", "O"]

        offsets = [2, 3, 5]
        original = ["B-ARG", "I-ARG", "O"]
        converted = _convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == ["O", "B-ARG", "I-ARG", "I-ARG", "O", "O", "O"]

    def test_wordpiece_tokenize_input(self):
        wordpieces, offsets, start_offsets = self.reader._wordpiece_tokenize_input(
            "This is a sentenceandsomepieces with a reallylongword".split(" ")
        )

        assert wordpieces == [
            "[CLS]",
            "this",
            "is",
            "a",
            "sentence",
            "##ands",
            "##ome",
            "##piece",
            "##s",
            "with",
            "a",
            "really",
            "##long",
            "##word",
            "[SEP]",
        ]
        assert [wordpieces[i] for i in offsets] == ["this", "is", "a", "##s", "with", "a", "##word"]
        assert [wordpieces[i] for i in start_offsets] == [
            "this",
            "is",
            "a",
            "sentence",
            "with",
            "a",
            "really",
        ]

    def test_read_from_file(self):
        conll_reader = self.reader
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / "conll_2012" / "subdomain")
        instances = ensure_list(instances)
        fields = instances[0].fields
        tokens = fields["metadata"]["words"]
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
        assert fields["verb_indicator"].labels[4] == 1

        assert fields["tags"].labels == [
            "O",
            "B-ARG0",
            "I-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "O",
            "O",
        ]

        fields = instances[1].fields
        tokens = fields["metadata"]["words"]
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
        assert fields["verb_indicator"].labels[10] == 1
        assert fields["tags"].labels == [
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "I-ARG1",
            "B-V",
            "B-ARG2",
            "O",
            "O",
        ]

        fields = instances[2].fields
        tokens = fields["metadata"]["words"]
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
        assert fields["verb_indicator"].labels[3] == 1
        assert fields["tags"].labels == [
            "O",
            "B-ARG0",
            "I-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "O",
            "O",
        ]

        fields = instances[3].fields
        tokens = fields["metadata"]["words"]
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
        assert fields["verb_indicator"].labels[12] == 1
        assert fields["tags"].labels == [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-V",
            "O",
            "O",
        ]

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields
        tokens = fields["metadata"]["words"]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["verb_indicator"].labels == [0, 0, 0, 0, 0, 0, 0]
        assert fields["tags"].labels == ["O", "O", "O", "O", "O", "O", "O"]

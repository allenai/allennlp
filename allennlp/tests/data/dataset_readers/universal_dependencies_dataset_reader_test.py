from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestUniversalDependenciesDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies.conllu"

    def test_read_from_file(self):
        reader = UniversalDependenciesDatasetReader()
        instances = list(reader.read(str(self.data_path)))

        instance = instances[0]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == [
            "What",
            "if",
            "Google",
            "Morphed",
            "Into",
            "GoogleOS",
            "?",
        ]

        assert fields["pos_tags"].labels == [
            "PRON",
            "SCONJ",
            "PROPN",
            "VERB",
            "ADP",
            "PROPN",
            "PUNCT",
        ]
        assert fields["head_tags"].labels == [
            "root",
            "mark",
            "nsubj",
            "advcl",
            "case",
            "obl",
            "punct",
        ]
        assert fields["head_indices"].labels == [0, 4, 4, 1, 6, 4, 4]

        instance = instances[1]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == [
            "What",
            "if",
            "Google",
            "expanded",
            "on",
            "its",
            "search",
            "-",
            "engine",
            "(",
            "and",
            "now",
            "e-mail",
            ")",
            "wares",
            "into",
            "a",
            "full",
            "-",
            "fledged",
            "operating",
            "system",
            "?",
        ]

        assert fields["pos_tags"].labels == [
            "PRON",
            "SCONJ",
            "PROPN",
            "VERB",
            "ADP",
            "PRON",
            "NOUN",
            "PUNCT",
            "NOUN",
            "PUNCT",
            "CCONJ",
            "ADV",
            "NOUN",
            "PUNCT",
            "NOUN",
            "ADP",
            "DET",
            "ADV",
            "PUNCT",
            "ADJ",
            "NOUN",
            "NOUN",
            "PUNCT",
        ]
        assert fields["head_tags"].labels == [
            "root",
            "mark",
            "nsubj",
            "advcl",
            "case",
            "nmod:poss",
            "compound",
            "punct",
            "compound",
            "punct",
            "cc",
            "advmod",
            "conj",
            "punct",
            "obl",
            "case",
            "det",
            "advmod",
            "punct",
            "amod",
            "compound",
            "obl",
            "punct",
        ]
        assert fields["head_indices"].labels == [
            0,
            4,
            4,
            1,
            15,
            15,
            9,
            9,
            15,
            9,
            13,
            13,
            9,
            15,
            4,
            22,
            22,
            20,
            20,
            22,
            22,
            4,
            4,
        ]

        instance = instances[2]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == [
            "[",
            "via",
            "Microsoft",
            "Watch",
            "from",
            "Mary",
            "Jo",
            "Foley",
            "]",
        ]
        assert fields["pos_tags"].labels == [
            "PUNCT",
            "ADP",
            "PROPN",
            "PROPN",
            "ADP",
            "PROPN",
            "PROPN",
            "PROPN",
            "PUNCT",
        ]
        assert fields["head_tags"].labels == [
            "punct",
            "case",
            "compound",
            "root",
            "case",
            "nmod",
            "flat",
            "flat",
            "punct",
        ]
        assert fields["head_indices"].labels == [4, 4, 4, 0, 6, 4, 6, 6, 4]

        # This instance tests specifically for filtering of elipsis:
        # https://universaldependencies.org/u/overview/specific-syntax.html#ellipsis
        # The original sentence is:
        # "Over 300 Iraqis are reported dead and 500 [reported] wounded in Fallujah alone."
        # But the second "reported" is elided, and as such isn't included in the syntax tree.
        instance = instances[3]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == [
            "Over",
            "300",
            "Iraqis",
            "are",
            "reported",
            "dead",
            "and",
            "500",
            "wounded",
            "in",
            "Fallujah",
            "alone",
            ".",
        ]
        assert fields["pos_tags"].labels == [
            "ADV",
            "NUM",
            "PROPN",
            "AUX",
            "VERB",
            "ADJ",
            "CCONJ",
            "NUM",
            "ADJ",
            "ADP",
            "PROPN",
            "ADV",
            "PUNCT",
        ]
        assert fields["head_tags"].labels == [
            "advmod",
            "nummod",
            "nsubj:pass",
            "aux:pass",
            "root",
            "xcomp",
            "cc",
            "conj",
            "orphan",
            "case",
            "obl",
            "advmod",
            "punct",
        ]
        assert fields["head_indices"].labels == [2, 3, 5, 5, 0, 5, 8, 5, 8, 11, 5, 11, 5]

    def test_read_from_file_with_language_specific_pos(self):
        reader = UniversalDependenciesDatasetReader(use_language_specific_pos=True)
        instances = list(reader.read(str(self.data_path)))

        instance = instances[0]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == [
            "What",
            "if",
            "Google",
            "Morphed",
            "Into",
            "GoogleOS",
            "?",
        ]

        assert fields["pos_tags"].labels == ["WP", "IN", "NNP", "VBD", "IN", "NNP", "."]
        assert fields["head_tags"].labels == [
            "root",
            "mark",
            "nsubj",
            "advcl",
            "case",
            "obl",
            "punct",
        ]
        assert fields["head_indices"].labels == [0, 4, 4, 1, 6, 4, 4]

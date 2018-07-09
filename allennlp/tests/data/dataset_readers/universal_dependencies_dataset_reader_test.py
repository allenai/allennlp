# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestUniversalDependenciesDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies.conllu"

    def test_read_from_file(self):
        reader = UniversalDependenciesDatasetReader(use_pos_tags=True)
        instances = list(reader.read(str(self.data_path)))

        instance = instances[0]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == ['ROOT_HEAD', 'What', 'if', 'Google',
                                                            'Morphed', 'Into', 'GoogleOS', '?']

        assert fields["pos_tags"].labels == ['ROOT_POS', 'PRON', 'SCONJ', 'PROPN', 'VERB', 'ADP', 'PROPN', 'PUNCT']
        assert fields["head_tags"].labels == ['root', 'root', 'mark', 'nsubj', 'advcl:if',
                                              'case', 'obl:into', 'punct']
        assert fields["head_indices"].labels == [0, 0, 4, 4, 1, 6, 4, 4]

        instance = instances[1]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == ['ROOT_HEAD', 'What', 'if', 'Google', 'expanded', 'on',
                                                            'its', 'search', '-', 'engine', '(', 'and',
                                                            'now', 'e-mail', ')', 'wares', 'into', 'a',
                                                            'full', '-', 'fledged', 'operating', 'system', '?']

        assert fields["pos_tags"].labels == ['ROOT_POS', 'PRON', 'SCONJ', 'PROPN', 'VERB', 'ADP', 'PRON', 'NOUN',
                                             'PUNCT', 'NOUN', 'PUNCT', 'CCONJ', 'ADV', 'NOUN', 'PUNCT', 'NOUN',
                                             'ADP', 'DET', 'ADV', 'PUNCT', 'ADJ', 'NOUN', 'NOUN', 'PUNCT']
        assert fields["head_tags"].labels == ['root', 'root', 'mark', 'nsubj', 'advcl:if', 'case', 'nmod:poss',
                                              'compound', 'punct', 'compound', 'punct', 'cc', 'advmod',
                                              'conj:and', 'punct', 'obl:on', 'case', 'det', 'advmod', 'punct',
                                              'amod', 'compound', 'obl:into', 'punct']
        assert fields["head_indices"].labels == [0, 0, 4, 4, 1, 15, 15, 9, 9, 15, 9, 13, 13,
                                                 9, 15, 4, 22, 22, 20, 20, 22, 22, 4, 4]

        instance = instances[2]
        fields = instance.fields
        assert [t.text for t in fields["words"].tokens] == ['ROOT_HEAD', '[', 'via', 'Microsoft', 'Watch',
                                                            'from', 'Mary', 'Jo', 'Foley', ']']
        assert fields["pos_tags"].labels == ['ROOT_POS', 'PUNCT', 'ADP', 'PROPN', 'PROPN', 'ADP',
                                             'PROPN', 'PROPN', 'PROPN', 'PUNCT']
        assert fields["head_tags"].labels == ['root', 'punct', 'case', 'compound', 'root', 'case',
                                              'nmod:from', 'flat', 'flat', 'punct']
        assert fields["head_indices"].labels == [0, 4, 4, 4, 0, 6, 4, 6, 6, 4]

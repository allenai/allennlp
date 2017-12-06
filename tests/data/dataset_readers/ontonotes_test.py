# pylint: disable=no-self-use,invalid-name

from nltk import Tree

from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.common.testing import AllenNlpTestCase


class TestOntonotes(AllenNlpTestCase):

    def test_dataset_iterator(self):
        reader = Ontonotes()
        annotated_sentences = list(reader.dataset_iterator('tests/fixtures/conll_2012/'))
        annotation = annotated_sentences[0]
        assert annotation.document_id == "test/test/01/test_001"
        assert annotation.sentence_id == 0
        assert annotation.words == ['Mali', 'government', 'officials', 'say', 'the', 'woman',
                                    "'s", 'confession', 'was', 'forced', '.']
        assert annotation.pos_tags == ['NNP', 'NN', 'NNS', 'VBP', 'DT',
                                       'NN', 'POS', 'NN', 'VBD', 'JJ', '.']
        assert annotation.word_senses == [None, None, 1, 1, None, 2, None, None, 1, None, None]
        assert annotation.predicate_framenet_ids == [None, None, None, '01', None,
                                                     None, None, None, '01', None, None]
        assert annotation.srl_frames == {"say": ['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                                                 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O'],
                                         "was": ['O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1', 'I-ARG1',
                                                 'I-ARG1', 'B-V', 'B-ARG2', 'O']}
        assert annotation.named_entities == ['B-GPE', 'O', 'O', 'O', 'O', 'O',
                                             'O', 'O', 'O', 'O', 'O']
        assert annotation.predicate_lemmas == [None, None, 'official', 'say', None,
                                               'man', None, None, 'be', None, None]
        assert annotation.speakers == [None, None, None, None, None, None,
                                       None, None, None, None, None]
        assert annotation.parse_tree == Tree.fromstring("(TOP(S(NP(NML Mali government) officials)"
                                                        "(VP say(SBAR(S(NP(NP the woman 's)"
                                                        " confession)(VP was(ADJP forced))))) .))")
        assert annotation.coref_spans == {(1, (4, 6)), (3, (4, 7))}

        annotation = annotated_sentences[1]
        assert annotation.document_id == "test/test/02/test_002"
        assert annotation.sentence_id == 0
        assert annotation.words == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month',
                                    'after', 'four', 'months', 'of', 'hearings', '.']
        assert annotation.pos_tags == ['DT', 'NN', 'VBD', 'PRP$', 'NN', 'JJ', 'NN',
                                       'IN', 'CD', 'NNS', 'IN', 'NNS', '.']
        assert annotation.word_senses == [None, 2, 5, None, 2, None, None,
                                          None, None, 1, None, 1, None]
        assert annotation.predicate_framenet_ids == [None, None, '01', None, None, None,
                                                     None, None, None, None, None, '01', None]
        assert annotation.srl_frames == {'rested': ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                                                    'I-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP',
                                                    'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
                                                    'I-ARGM-TMP', 'I-ARGM-TMP', 'O'],
                                         'hearings': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                                      'O', 'O', 'O', 'B-V', 'O']}
        assert annotation.named_entities == ['O', 'O', 'O', 'O', 'O', 'B-DATE', 'I-DATE',
                                             'O', 'B-DATE', 'I-DATE', 'O', 'O', 'O']
        assert annotation.predicate_lemmas == [None, 'prosecution', 'rest', None, 'case',
                                               None, None, None, None, 'month', None, 'hearing', None]
        assert annotation.speakers == [None, None, None, None, None, None,
                                       None, None, None, None, None, None, None]
        assert annotation.parse_tree == Tree.fromstring("(TOP (S (NP The prosecution)"
                                                        " (VP rested (NP its case) (NP last month)"
                                                        " (PP after (NP (NP four months) (PP of "
                                                        "(NP hearings))))) .)) ")
        assert annotation.coref_spans == {(2, (0, 1)), (2, (3, 3))}

        annotation = annotated_sentences[2]
        assert annotation.document_id == 'test/test/03/test_003'
        assert annotation.sentence_id == 0
        assert annotation.words == ['Denise', 'Dillon', 'Headline', 'News', '.']
        assert annotation.pos_tags == ['NNP', 'NNP', 'NNP', 'NNP', '.']
        assert annotation.word_senses == [None, None, None, None, None]
        assert annotation.predicate_framenet_ids == [None, None, None, None, None]
        assert annotation.srl_frames == {}
        assert annotation.named_entities == ['B-PERSON', 'I-PERSON',
                                             'B-WORK_OF_ART', 'I-WORK_OF_ART', 'O']
        assert annotation.predicate_lemmas == [None, None, None, None, None]
        assert annotation.speakers == [None, None, None, None, None]
        assert annotation.parse_tree == Tree.fromstring('(TOP (FRAG (NP Denise Dillon)'
                                                        ' (NP Headline News) .))')
        assert annotation.coref_spans == {(2, (0, 1))}

    def test_dataset_path_iterator(self):
        reader = Ontonotes()
        files = list(reader.dataset_path_iterator('tests/fixtures/conll_2012/'))
        assert files == ['tests/fixtures/conll_2012/example.gold_conll']

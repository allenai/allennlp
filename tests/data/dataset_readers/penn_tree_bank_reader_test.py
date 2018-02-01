# pylint: disable=no-self-use,invalid-name,protected-access

from nltk.tree import Tree

from allennlp.data.dataset_readers import PennTreeBankConstituencySpanDatasetReader
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans


class TestPennTreeBankConstituencySpanReader(AllenNlpTestCase):

    def setUp(self):
        super(TestPennTreeBankConstituencySpanReader, self).setUp()
        self.span_width = 5

    def test_read_from_file(self):

        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        instances = ptb_reader.read('tests/fixtures/data/example_ptb.trees')

        assert len(instances) == 2

        fields = instances[0].fields
        tokens = [x.text for x in fields["tokens"].tokens]
        pos_tags = fields["pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields["spans"].field_list]
        span_labels = fields["span_labels"].labels

        assert tokens == ['Also', ',', 'because', 'UAL', 'Chairman', 'Stephen', 'Wolf',
                          'and', 'other', 'UAL', 'executives', 'have', 'joined', 'the',
                          'pilots', "'", 'bid', ',', 'the', 'board', 'might', 'be', 'forced',
                          'to', 'exclude', 'him', 'from', 'its', 'deliberations', 'in',
                          'order', 'to', 'be', 'fair', 'to', 'other', 'bidders', '.']
        assert pos_tags == ['RB', ',', 'IN', 'NNP', 'NNP', 'NNP', 'NNP', 'CC', 'JJ', 'NNP',
                            'NNS', 'VBP', 'VBN', 'DT', 'NNS', 'POS', 'NN', ',', 'DT', 'NN',
                            'MD', 'VB', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'PRP$',
                            'NNS', 'IN', 'NN', 'TO', 'VB', 'JJ', 'TO', 'JJ', 'NNS', '.']

        assert spans == enumerate_spans(tokens)
        gold_tree = Tree.fromstring("(VROOT(S(ADVP(RB Also))(, ,)(SBAR-PRP(IN because)"
                                    "(S(NP-SBJ(NP(NNP UAL)(NNP Chairman)(NNP Stephen)(NNP Wolf))"
                                    "(CC and)(NP(JJ other)(NNP UAL)(NNS executives)))(VP(VBP have)"
                                    "(VP(VBN joined)(NP(NP(DT the)(NNS pilots)(POS '))(NN bid))))))"
                                    "(, ,)(NP-SBJ(DT the)(NN board))(VP(MD might)(VP(VB be)(VP(VBN "
                                    "forced)(S(VP(TO to)(VP(VB exclude)(NP(PRP him))(PP-CLR(IN from)"
                                    "(NP(PRP$ its)(NNS deliberations)))(SBAR-PRP(IN in)(NN order)(S("
                                    "VP(TO to)(VP(VB be)(ADJP-PRD(JJ fair)(PP(TO to)(NP(JJ other)(NNS "
                                    "bidders))))))))))))))(. .)))")

        correct_spans_and_labels = {}
        ptb_reader._get_gold_spans(gold_tree, 0, correct_spans_and_labels)
        for span, label in zip(spans, span_labels):
            if label != "NO-LABEL":
                assert correct_spans_and_labels[span] == label


        fields = instances[1].fields
        tokens = [x.text for x in fields["tokens"].tokens]
        pos_tags = fields["pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields["spans"].field_list]
        span_labels = fields["span_labels"].labels

        assert tokens == ['That', 'could', 'cost', 'him', 'the', 'chance',
                          'to', 'influence', 'the', 'outcome', 'and', 'perhaps',
                          'join', 'the', 'winning', 'bidder', '.']

        assert pos_tags == ['DT', 'MD', 'VB', 'PRP', 'DT', 'NN',
                            'TO', 'VB', 'DT', 'NN', 'CC', 'RB', 'VB', 'DT',
                            'VBG', 'NN', '.']

        assert spans == enumerate_spans(tokens)

        gold_tree = Tree.fromstring("(VROOT(S(NP-SBJ(DT That))(VP(MD could)(VP(VB cost)(NP(PRP him))"
                                    "(NP(DT the)(NN chance)(S(VP(TO to)(VP(VP(VB influence)(NP(DT the)"
                                    "(NN outcome)))(CC and)(VP(ADVP(RB perhaps))(VB join)(NP(DT the)"
                                    "(VBG winning)(NN bidder)))))))))(. .)))")

        correct_spans_and_labels = {}
        ptb_reader._get_gold_spans(gold_tree, 0, correct_spans_and_labels)
        for span, label in zip(spans, span_labels):
            if label != "NO-LABEL":
                assert correct_spans_and_labels[span] == label

    def test_get_gold_spans_correctly_extracts_spans(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        tree = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")

        span_dict = {}
        ptb_reader._get_gold_spans(tree, 0, span_dict)
        spans = list(span_dict.items()) # pylint: disable=protected-access
        assert spans == [((0, 0), 'D-POS'), ((1, 1), 'N-POS'), ((0, 1), 'NP'),
                         ((2, 2), 'V-POS'), ((3, 3), 'D-POS'), ((4, 4), 'N-POS'),
                         ((3, 4), 'NP'), ((2, 4), 'VP'), ((0, 4), 'S')]

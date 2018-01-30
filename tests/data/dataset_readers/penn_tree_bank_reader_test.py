# pylint: disable=no-self-use,invalid-name
from collections import OrderedDict

from nltk.tree import Tree

from allennlp.data.dataset_readers import PennTreeBankDatasetReader
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans


class TestPennTreeBankReader(AllenNlpTestCase):

    def setUp(self):
        super(TestPennTreeBankReader, self).setUp()
        self.span_width = 5

    def test_read_from_file(self):

        ptb_reader = PennTreeBankDatasetReader()
        dataset = ptb_reader.read('tests/fixtures/data/example_ptb.trees')

        assert len(dataset.instances) == 2
        instances = dataset.instances

        fields = instances[0].fields
        tokens = [x.text for x in fields["tokens"].tokens]
        pos_tags = fields["pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields["spans"].field_list]

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

        fields = instances[1].fields
        tokens = [x.text for x in fields["tokens"].tokens]
        pos_tags = fields["pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields["spans"].field_list]


        assert tokens == ['That', 'could', 'cost', 'him', 'the', 'chance',
                          'to', 'influence', 'the', 'outcome', 'and', 'perhaps',
                          'join', 'the', 'winning', 'bidder', '.']

        assert pos_tags == ['DT', 'MD', 'VB', 'PRP', 'DT', 'NN',
                            'TO', 'VB', 'DT', 'NN', 'CC', 'RB', 'VB', 'DT',
                            'VBG', 'NN', '.']

        assert spans == enumerate_spans(tokens)

    def test_get_gold_spans_correctly_extracts_spans(self):

        ptb_reader = PennTreeBankDatasetReader()
        tree = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        spans = list(ptb_reader._get_gold_spans(tree, 0, OrderedDict()).items()) # pylint: disable=protected-access
        assert spans == [((0, 0), 'D-POS'), ((1, 1), 'N-POS'), ((0, 1), 'NP'),
                         ((2, 2), 'V-POS'), ((3, 3), 'D-POS'), ((4, 4), 'N-POS'),
                         ((3, 4), 'NP'), ((2, 4), 'VP'), ((0, 4), 'S')]
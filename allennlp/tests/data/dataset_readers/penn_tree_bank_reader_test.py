from nltk.tree import Tree

from allennlp.data.dataset_readers import PennTreeBankConstituencySpanDatasetReader
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans


class TestPennTreeBankConstituencySpanReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.span_width = 5

    def test_read_from_file(self):

        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        instances = ptb_reader.read(str(self.FIXTURES_ROOT / "data" / "example_ptb.trees"))

        assert len(instances) == 2

        fields = instances[0].fields
        tokens = [x.text for x in fields["tokens"].tokens]
        pos_tags = fields["pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields["spans"].field_list]
        span_labels = fields["span_labels"].labels

        assert tokens == [
            "Also",
            ",",
            "because",
            "UAL",
            "Chairman",
            "Stephen",
            "Wolf",
            "and",
            "other",
            "UAL",
            "executives",
            "have",
            "joined",
            "the",
            "pilots",
            "'",
            "bid",
            ",",
            "the",
            "board",
            "might",
            "be",
            "forced",
            "to",
            "exclude",
            "him",
            "from",
            "its",
            "deliberations",
            "in",
            "order",
            "to",
            "be",
            "fair",
            "to",
            "other",
            "bidders",
            ".",
        ]
        assert pos_tags == [
            "RB",
            ",",
            "IN",
            "NNP",
            "NNP",
            "NNP",
            "NNP",
            "CC",
            "JJ",
            "NNP",
            "NNS",
            "VBP",
            "VBN",
            "DT",
            "NNS",
            "POS",
            "NN",
            ",",
            "DT",
            "NN",
            "MD",
            "VB",
            "VBN",
            "TO",
            "VB",
            "PRP",
            "IN",
            "PRP$",
            "NNS",
            "IN",
            "NN",
            "TO",
            "VB",
            "JJ",
            "TO",
            "JJ",
            "NNS",
            ".",
        ]

        assert spans == enumerate_spans(tokens)
        gold_tree = Tree.fromstring(
            "(S(ADVP(RB Also))(, ,)(SBAR(IN because)"
            "(S(NP(NP(NNP UAL)(NNP Chairman)(NNP Stephen)(NNP Wolf))"
            "(CC and)(NP(JJ other)(NNP UAL)(NNS executives)))(VP(VBP have)"
            "(VP(VBN joined)(NP(NP(DT the)(NNS pilots)(POS '))(NN bid))))))"
            "(, ,)(NP(DT the)(NN board))(VP(MD might)(VP(VB be)(VP(VBN "
            "forced)(S(VP(TO to)(VP(VB exclude)(NP(PRP him))(PP(IN from)"
            "(NP(PRP$ its)(NNS deliberations)))(SBAR(IN in)(NN order)(S("
            "VP(TO to)(VP(VB be)(ADJP(JJ fair)(PP(TO to)(NP(JJ other)(NNS "
            "bidders))))))))))))))(. .))"
        )

        assert fields["metadata"].metadata["gold_tree"] == gold_tree
        assert fields["metadata"].metadata["tokens"] == tokens

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

        assert tokens == [
            "That",
            "could",
            "cost",
            "him",
            "the",
            "chance",
            "to",
            "influence",
            "the",
            "outcome",
            "and",
            "perhaps",
            "join",
            "the",
            "winning",
            "bidder",
            ".",
        ]

        assert pos_tags == [
            "DT",
            "MD",
            "VB",
            "PRP",
            "DT",
            "NN",
            "TO",
            "VB",
            "DT",
            "NN",
            "CC",
            "RB",
            "VB",
            "DT",
            "VBG",
            "NN",
            ".",
        ]

        assert spans == enumerate_spans(tokens)

        gold_tree = Tree.fromstring(
            "(S(NP(DT That))(VP(MD could)(VP(VB cost)(NP(PRP him))"
            "(NP(DT the)(NN chance)(S(VP(TO to)(VP(VP(VB influence)(NP(DT the)"
            "(NN outcome)))(CC and)(VP(ADVP(RB perhaps))(VB join)(NP(DT the)"
            "(VBG winning)(NN bidder)))))))))(. .))"
        )

        assert fields["metadata"].metadata["gold_tree"] == gold_tree
        assert fields["metadata"].metadata["tokens"] == tokens

        correct_spans_and_labels = {}
        ptb_reader._get_gold_spans(gold_tree, 0, correct_spans_and_labels)
        for span, label in zip(spans, span_labels):
            if label != "NO-LABEL":
                assert correct_spans_and_labels[span] == label

    def test_strip_functional_tags(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        # Get gold spans should strip off all the functional tags.
        tree = Tree.fromstring(
            "(S (NP=PRP (D the) (N dog)) (VP-0 (V chased) (NP|FUN-TAGS (D the) (N cat))))"
        )
        ptb_reader._strip_functional_tags(tree)
        assert tree == Tree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))"
        )

    def test_get_gold_spans_correctly_extracts_spans(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        tree = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")

        span_dict = {}
        ptb_reader._get_gold_spans(tree, 0, span_dict)
        spans = list(span_dict.items())
        assert spans == [((0, 1), "NP"), ((3, 4), "NP"), ((2, 4), "VP"), ((0, 4), "S")]

    def test_get_gold_spans_correctly_extracts_spans_with_nested_labels(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        # Here we have a parse with several nested labels - particularly the (WHNP (WHNP (WP What)))
        # fragment. These should be concatenated into a single label by get_gold_spans.
        tree = Tree.fromstring(
            """
            (S
        (` ``)
        (S-TPC
        (NP-SBJ (PRP We))
        (VP
            (VBP have)
            (S
            (VP
                (TO to)
                (VP
                (VP
                    (VB clear)
                    (PRT (RP up))
                    (NP (DT these) (NNS issues)))
                (CC and)
                (VP
                    (VB find)
                    (PRT (RP out))
                    (SBAR-NOM
                    (WHNP (WHNP (WP what)))
                    (S
                        (VP
                        (VBZ is)
                        (ADJP-PRD (JJ present))
                        (SBAR
                            (WHNP (WDT that))
                            (S
                            (VP
                                (VBZ is)
                                (VP
                                (VBG creating)
                                (NP (JJ artificial) (NN volatility)))))))))))))))
        (, ,)
        ('' '')
        (NP-SBJ (NNP Mr.) (NNP Fisher))
        (VP (VBD said))
        (. .))
        """
        )
        span_dict = {}
        ptb_reader._strip_functional_tags(tree)
        ptb_reader._get_gold_spans(tree, 0, span_dict)
        assert span_dict == {
            (1, 1): "NP",
            (5, 5): "PRT",
            (6, 7): "NP",
            (4, 7): "VP",
            (10, 10): "PRT",
            (11, 11): "WHNP-WHNP",
            (13, 13): "ADJP",
            (14, 14): "WHNP",
            (17, 18): "NP",
            (16, 18): "VP",
            (15, 18): "S-VP",
            (14, 18): "SBAR",
            (12, 18): "S-VP",
            (11, 18): "SBAR",
            (9, 18): "VP",
            (4, 18): "VP",
            (3, 18): "S-VP",
            (2, 18): "VP",
            (1, 18): "S",
            (21, 22): "NP",
            (23, 23): "VP",
            (0, 24): "S",
        }

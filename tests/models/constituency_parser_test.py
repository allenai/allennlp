# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing.model_test_case import ModelTestCase

class SpanConstituencyParserTest(ModelTestCase):

    def setUp(self):
        super(SpanConstituencyParserTest, self).setUp()
        self.set_up_model("tests/fixtures/constituency_parser/constituency_parser.json",
                          "tests/fixtures/data/example_ptb.trees")

    def test_span_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_decode_runs(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)
        assert set(decode_output_dict.keys()) == {'spans', 'class_probabilities', 'trees',
                                                  'tokens', 'token_mask', 'loss'}

    def test_resolve_overlap_conflicts_greedily(self):
        spans = [{"start": 1, "end": 5, "no_label_prob": 0.7, "label_prob": 0.2},
                 {"start": 2, "end": 7, "no_label_prob": 0.5, "label_prob": 0.3}]
        resolved_spans = self.model.resolve_overlap_conflicts_greedily(spans)
        assert resolved_spans == [{"start": 2, "end": 7, "no_label_prob": 0.5, "label_prob": 0.3}]

    def test_construct_tree_from_spans(self):
        # (S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))
        tree_spans = [((0, 1), 'D-POS'), ((1, 2), 'N-POS'), ((0, 2), 'NP'),
                      ((2, 3), 'V-POS'), ((3, 4), 'D-POS'), ((4, 5), 'N-POS'),
                      ((3, 5), 'NP'), ((2, 5), 'VP'), ((0, 5), 'S')]
        sentence = ["the", "dog", "chased", "the", "cat"]
        tree = self.model.construct_tree_from_spans({x:y for x, y in tree_spans}, sentence)
        # pylint: disable=bad-continuation
        correct_tree = {
                "label": "S",
                "start": 0,
                "end": 5,
                "children": [
                        {
                            "label": "NP",
                            "start": 0,
                            "end": 2,
                            "children": [
                                {
                                    "label": "D-POS",
                                    "start": 0,
                                    "end": 1,
                                    "word": "the",
                                    "is_leaf": True
                                },
                                {
                                    "label": "N-POS",
                                    "start": 1,
                                    "end": 2,
                                    "word": "dog",
                                    "is_leaf": True
                                }
                            ]
                        },
                        {
                            "label": "VP",
                            "start": 2,
                            "end": 5,
                            "children": [
                                {
                                    "label": "V-POS",
                                    "start": 2,
                                    "end": 3,
                                    "word": "chased",
                                    "is_leaf": True
                                },
                                {
                                    "label": "NP",
                                    "start": 3,
                                    "end": 5,
                                    "children": [
                                        {
                                            "label": "D-POS",
                                            "start": 3,
                                            "end": 4,
                                            "word": "the",
                                            "is_leaf": True
                                        },
                                        {
                                            "label": "N-POS",
                                            "start": 4,
                                            "end": 5,
                                            "word": "cat",
                                            "is_leaf": True
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
        # pylint: enable=bad-continuation
        assert tree == correct_tree

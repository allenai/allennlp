# pylint: disable=invalid-name,no-self-use
from allennlp.common.testing import ModelTestCase
from allennlp.models import WikiTablesSemanticParser


class WikiTablesSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(WikiTablesSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/wikitables_semantic_parser/experiment.json",
                          "tests/fixtures/data/wikitables/sample_data.examples")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_get_unique_elements(self):
        # pylint: disable=protected-access
        production_rules = [
                # We won't bother with constructing the last element of the ProductionRuleArray
                # here, the Dict[str, torch.Tensor].  It's not necessary for this test.  We'll just
                # give each element a unique index that we can check in the resulting dictionaries.
                # arrays.
                [{"left": ('r', True, 1), "right": ('d', True, 2)},
                 {"left": ('r', True, 1), "right": ('c', True, 3)},
                 {"left": ('d', True, 2), "right": ('entity_1', False, 4)}],
                [{"left": ('r', True, 1), "right": ('d', True, 2)},
                 {"left": ('d', True, 2), "right": ('entity_2', False, 5)},
                 {"left": ('d', True, 2), "right": ('entity_1', False, 4)},
                 {"left": ('d', True, 2), "right": ('entity_3', False, 6)}]
                ]
        nonterminals, terminals = WikiTablesSemanticParser._get_unique_elements(production_rules)
        assert nonterminals == {
                'r': 1,
                'd': 2,
                'c': 3,
                }
        assert terminals == {
                'entity_1': 4,
                'entity_2': 5,
                'entity_3': 6,
                }

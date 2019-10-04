from allennlp.common.testing import ModelTestCase
from allennlp.state_machines.states import GrammarStatelet
from allennlp.models.semantic_parsing.text2sql_parser import Text2SqlParser
from allennlp.semparse.worlds.text2sql_world import Text2SqlWorld


class Text2SqlParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        self.set_up_model(
            str(self.FIXTURES_ROOT / "semantic_parsing" / "text2sql" / "experiment.json"),
            str(self.FIXTURES_ROOT / "data" / "text2sql" / "restaurants_tiny.json"),
        )
        self.schema = str(self.FIXTURES_ROOT / "data" / "text2sql" / "restaurants-schema.csv")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_grammar_statelet(self):
        valid_actions = None
        world = Text2SqlWorld(self.schema)

        sql = ["SELECT", "COUNT", "(", "*", ")", "FROM", "LOCATION", ",", "RESTAURANT", ";"]
        action_sequence, valid_actions = world.get_action_sequence_and_all_actions(sql)

        grammar_state = GrammarStatelet(
            ["statement"], valid_actions, Text2SqlParser.is_nonterminal, reverse_productions=True
        )
        for action in action_sequence:
            grammar_state = grammar_state.take_action(action)
        assert grammar_state._nonterminal_stack == []

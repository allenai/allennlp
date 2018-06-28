# pylint: disable=no-self-use,invalid-name
from typing import List

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds import RateCalculusWorld
from allennlp.semparse.contexts.question_knowledge_graph import QuestionKnowledgeGraph
from allennlp.data.tokenizers import Token


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestRateCalculusWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        question = ['I', 'have', 'a', '20', 'Dollar', 'bill', 'and', 'a', '50', 'Dollar', 'bill']
        question_tokens = [Token(x) for x in question]
        question_knowledge_graph = QuestionKnowledgeGraph.read(question_tokens)
        self.world = RateCalculusWorld(question_knowledge_graph)

    def test_world_parses_equality(self):
        sempre_form = "(Equals 50 20)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "E(num:50,num:20)"

    def test_world_parses_values(self):
        sempre_form = "(Value o1 Dollar)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "V(O1,U5)"

    def test_world_parses_rates(self):
        sempre_form = "(Rate o1 Dollar Unit)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "R(O1,U5,U6)"

    def test_world_parses_value_constraint(self):
        sempre_form = "(Equals (Value o1 Dollar) 20)"
        expression = self.world.parse_logical_form(sempre_form)
        print("ACTION SEQ: ", self.world.get_action_sequence(expression))
        assert str(expression) == "E(V(O1,U5),num:20)"

    def test_world_parses_rate_constraint(self):
        sempre_form = "(Equals (Rate o1 Dollar Unit) 20)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "E(R(O1,U5,U6),num:20)"

    def test_world_parses_union_constraint(self):
        sempre_form = "(IsPart o1 o0)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "J(O1,O0)"

    def test_world_parses_conjunction(self):
        sempre_form = "(And (Equals 20 20) (Equals 50 50))"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "A(E(num:20,num:20),E(num:50,num:50))"

    def test_world_parses_nested_conjunction(self):
        sempre_form = "(And (Equals x1 x2) (And (Equals 20 20) (Equals 50 50)))"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "A(E(X1,X2),A(E(num:20,num:20),E(num:50,num:50)))"

    def test_get_valid_actions_returns_correct_set(self):
        # This test checks valid actions for each type match
        valid_actions = self.world.get_valid_actions()

        assert set(valid_actions.keys()) == {
                '@start@',
                'b',
                'd',
                'n',
                'o',
                '<o,<o,b>>',
                '<o,<d,<d,n>>>',
                '<o,<d,n>>',
                '<b,<b,b>>',
                '<n,<n,b>>',
                '<n,<n,n>>'
                }

        check_productions_match(valid_actions['@start@'],
                                ['b'])

        check_productions_match(valid_actions['b'],
                                ['[<b,<b,b>>, b, b]', '[<n,<n,b>>, n, n]', '[<o,<o,b>>, o, o]'])

        check_productions_match(valid_actions['d'],
                                ['Dollar', 'Unit', 'Unit0', 'Unit1'])

        number_productions = ['20',
                              '50',
                              '[<o,<d,<d,n>>>, o, d, d]',
                              '[<o,<d,n>>, o, d]',
                              'x0',
                              'x1',
                              'q0',
                              'q1',
                              '1',
                              '[<n,<n,n>>, n, n]']
        check_productions_match(valid_actions['n'], number_productions)

        check_productions_match(valid_actions['o'],
                                ['o0', 'o1', 'o2', 'o3', 'o4', 'o5'])

        check_productions_match(valid_actions['<o,<d,n>>'],
                                ['Value'])

        check_productions_match(valid_actions['<o,<d,<d,n>>>'],
                                ['Rate'])

        check_productions_match(valid_actions['<o,<o,b>>'],
                                ['IsPart'])

        check_productions_match(valid_actions['<n,<n,b>>'],
                                ['Equals'])

        check_productions_match(valid_actions['<b,<b,b>>'],
                                ['And'])

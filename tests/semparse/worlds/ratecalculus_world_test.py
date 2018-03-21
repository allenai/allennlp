# pylint: disable=no-self-use,invalid-name
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse import ParsingError
from allennlp.data.semparse.worlds import RateCalculusWorld
from allennlp.data.tokenizers import Token


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestRateCalculusWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        question_tokens = [Token(x) for x in ['$20', 'unit', 'dollar', 'x', 'y', '50km', '?']]
        self.world = RateCalculusWorld(question_tokens)

    def test_world_parses_equality(self):
        sempre_form = "(Equals 50 20)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "E(num:50,num:20)"

    def test_world_parses_values(self):
        sempre_form = "(Value x dollar)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "V1(X,D)"

    def test_world_parses_rates(self):
        sempre_form = "(Rate x dollar unit)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "R1(X,D,U)"

    def test_world_parses_value_constraint(self):
        sempre_form = "(Equals (Value x dollar) 20)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "E(V1(X,D),num:20)"

    def test_world_parses_rate_constraint(self):
        sempre_form = "(Equals (Rate x dollar unit) 20)"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "E(R1(X,D,U),num:20)"

    def test_world_parses_conjunction(self):
        sempre_form = "(And (Equals 20 20) (Equals 50 50))"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "A(E(num:20,num:20),E(num:50,num:50))"

    def test_world_parses_nested_conjunction(self):
        sempre_form = "(And (Equals p q) (And (Equals 20 20) (Equals 50 50)))"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "A(E(P,Q),A(E(num:20,num:20),E(num:50,num:50)))"

    def test_get_valid_actions_returns_correct_set(self):
        # This test checks valid actions for each type match
        valid_actions = self.world.get_valid_actions()

        assert set(valid_actions.keys()) == {
                '@START@',
                'b',
                'd',
                'n',
                '<o,<d,<d,n>>>',
                '<o,<d,n>>',
                '<b,<b,b>>',
                '<n,<n,b>>'
                }

        check_productions_match(valid_actions['@START@'],
                                ['b'])

        check_productions_match(valid_actions['b'],
                                ['[<b,<b,b>>, b, b]', '[<n,<n,b>>, n, n]'])

        check_productions_match(valid_actions['d'],
                                ['dollar', 'unit'])

        check_productions_match(valid_actions['n'],
                                ['num:20', 'num:50', '[<o,<d,<d,n>>>, o, d, d]', '[<o,<d,n>>, o, d]', 'p', 'q'])

        check_productions_match(valid_actions['<o,<d,<d,n>>>'],
                                ['Rate'])

        check_productions_match(valid_actions['<o,<d,n>>'],
                                ['Value'])

        check_productions_match(valid_actions['<n,<n,b>>'],
                                ['Equals'])

        check_productions_match(valid_actions['<b,<b,b>>'],
                                ['And'])
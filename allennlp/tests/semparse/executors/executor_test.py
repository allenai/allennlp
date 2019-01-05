# pylint: disable=no-self-use,invalid-name,protected-access
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import Executor, ExecutionError, ParsingError, predicate
from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType

class ArithmeticExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.start_types = {int}

    @predicate
    def add(self, num1: int, num2: int) -> int:
        return num1 + num2

    @predicate
    def sum(self, numbers: List[int]) -> int:
        return sum(numbers)

    @predicate
    def subtract(self, num1: int, num2: int) -> int:
        return num1 - num2

    @predicate
    def power(self, num1: int, num2: int) -> int:
        return num1 ** num2

    @predicate
    def multiply(self, num1: int, num2: int) -> int:
        return num1 * num2

    @predicate
    def divide(self, num1: int, num2: int) -> int:
        return num1 / num2

    @predicate
    def halve(self, num1: int) -> int:
        return num1 / 2

    @predicate
    def three(self) -> int:
        return 3

    def not_a_predicate(self) -> int:
        return 5


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class ExecutorTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.executor = ArithmeticExecutor()

    def test_constant_logical_form(self):
        assert self.executor.execute('5') == 5
        assert self.executor.execute('0.2') == 0.2
        assert self.executor.execute('1.2e-2') == 0.012
        assert self.executor.execute('"5"') == '5'
        assert self.executor.execute('string') == 'string'
        assert self.executor.execute('"string"') == 'string'

    def test_error_message_with_wrong_arguments(self):
        with pytest.raises(ExecutionError):
            self.executor.execute('add')
        with pytest.raises(ExecutionError):
            self.executor.execute('(add)')
        with pytest.raises(ExecutionError):
            self.executor.execute('(add 2)')
        # If they are explicitly marked as strings, that's ok.
        assert self.executor.execute('"add"') == 'add'

    def test_not_all_functions_are_predicates(self):
        # This should not execute to 5, but instead be treated as a constant.
        self.executor.execute('not_a_predicate') == 'not_a_predicate'

    def test_basic_logical_form(self):
        assert self.executor.execute('three') == 3
        assert self.executor.execute('(add 2 3)') == 5
        assert self.executor.execute('(subtract 2 3)') == -1
        assert self.executor.execute('(halve 20)') == 10

    def test_list_types(self):
        assert self.executor.execute('(sum (2))') == 2
        assert self.executor.execute('(sum (2 3))') == 5
        assert self.executor.execute('(sum (2 3 10 -2 -5))') == 8
        assert self.executor.execute('(sum (2 three (halve 4) (add -5 -2)))') == 0

    def test_nested_logical_form(self):
        assert self.executor.execute('(add 2 (subtract 4 2))') == 4
        assert self.executor.execute('(halve (multiply (divide 9 3) (power 2 3)))') == 12

    def test_type_inference(self):
        assert str(self.executor._name_mapper.get_signature('add')) == '<i,<i,i>>'
        assert str(self.executor._name_mapper.get_signature('subtract')) == '<i,<i,i>>'
        assert str(self.executor._name_mapper.get_signature('power')) == '<i,<i,i>>'
        assert str(self.executor._name_mapper.get_signature('multiply')) == '<i,<i,i>>'
        assert str(self.executor._name_mapper.get_signature('divide')) == '<i,<i,i>>'
        assert str(self.executor._name_mapper.get_signature('halve')) == '<i,i>'
        assert str(self.executor._name_mapper.get_signature('three')) == 'i'
        assert str(self.executor._name_mapper.get_signature('sum')) == '<l,i>'

    def test_get_valid_actions(self):
        valid_actions = self.executor.get_valid_actions()
        assert valid_actions.keys() == {'@start@', '<i,<i,i>>', '<i,i>', '<l,i>', 'i'}
        check_productions_match(valid_actions['@start@'],
                                ['i'])
        check_productions_match(valid_actions['<i,<i,i>>'],
                                ['add', 'subtract', 'multiply', 'divide', 'power'])
        check_productions_match(valid_actions['<i,i>'],
                                ['halve'])
        check_productions_match(valid_actions['<l,i>'],
                                ['sum'])
        check_productions_match(valid_actions['i'],
                                ['[<i,<i,i>>, i, i]', '[<i,i>, i]', '[<l,i>, l]', 'three'])

    def test_logical_form_to_action_sequence(self):
        action_sequence = self.executor.logical_form_to_action_sequence('(add 2 3)')
        assert action_sequence == ['@start@ -> i',
                                   'i -> [<i,<i,i>>, i, i]',
                                   '<i,<i,i>> -> add',
                                   'i -> 2',
                                   'i -> 3']

        action_sequence = self.executor.logical_form_to_action_sequence('(halve (subtract 8 three))')
        assert action_sequence == ['@start@ -> i',
                                   'i -> [<i,i>, i]',
                                   '<i,i> -> halve',
                                   'i -> [<i,<i,i>>, i, i]',
                                   '<i,<i,i>> -> subtract',
                                   'i -> 8',
                                   'i -> three']

        logical_form = '(halve (multiply (divide 9 three) (power 2 3)))'
        action_sequence = self.executor.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> i',
                                   'i -> [<i,i>, i]',
                                   '<i,i> -> halve',
                                   'i -> [<i,<i,i>>, i, i]',
                                   '<i,<i,i>> -> multiply',
                                   'i -> [<i,<i,i>>, i, i]',
                                   '<i,<i,i>> -> divide',
                                   'i -> 9',
                                   'i -> three',
                                   'i -> [<i,<i,i>>, i, i]',
                                   '<i,<i,i>> -> power',
                                   'i -> 2',
                                   'i -> 3']

    def test_action_sequence_to_logical_form(self):
        logical_form = '(add 2 3)'
        action_sequence = self.executor.logical_form_to_action_sequence(logical_form)
        recovered_logical_form = self.executor.action_sequence_to_logical_form(action_sequence)
        assert recovered_logical_form == logical_form

        logical_form = '(halve (multiply (divide 9 three) (power 2 3)))'
        action_sequence = self.executor.logical_form_to_action_sequence(logical_form)
        recovered_logical_form = self.executor.action_sequence_to_logical_form(action_sequence)
        assert recovered_logical_form == logical_form

    def test_logical_form_parsing_fails_on_bad_inputs(self):
        # We don't catch all type inconsistencies in the code, but we _do_ catch some.  If we add
        # more that we catch, this is a good place to test for them.
        with pytest.raises(ParsingError, match='Wrong number of arguments'):
            self.executor.logical_form_to_action_sequence('(halve 2 3)')
        with pytest.raises(ParsingError, match='Wrong number of arguments'):
            self.executor.logical_form_to_action_sequence('(add 3)')
        with pytest.raises(ParsingError, match='Constant expressions not implemented yet'):
            self.executor.logical_form_to_action_sequence('add')
        with pytest.raises(ParsingError, match='Bare lists not implemented yet'):
            self.executor.logical_form_to_action_sequence('(sum (3 2))')
        with pytest.raises(ParsingError, match='did not have expected type'):
            self.executor.logical_form_to_action_sequence('(sum (add 2 3))')

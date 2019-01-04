# pylint: disable=no-self-use,invalid-name
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import Executor, ExecutionError, predicate

class ArithmeticExecutor(Executor):
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

    def test_nested_logical_form(self):
        assert self.executor.execute('(add 2 (subtract 4 2))') == 4
        assert self.executor.execute('(halve (multiply (divide 9 3) (power 2 3)))') == 12

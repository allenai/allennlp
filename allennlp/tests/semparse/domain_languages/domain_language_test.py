# pylint: disable=no-self-use,invalid-name,protected-access
from typing import Callable, List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import ExecutionError, ParsingError
from allennlp.semparse import DomainLanguage, predicate, predicate_with_side_args

class Arithmetic(DomainLanguage):
    def __init__(self):
        super().__init__(start_types={int}, allowed_constants={
                # We unfortunately have to explicitly enumerate all allowed constants in the
                # grammar.  Because we'll be inducing a grammar for this language for use with a
                # semantic parser, we need the grammar to be finite, which means we can't allow
                # arbitrary constants (you can't parameterize an infinite categorical
                # distribution).  So our Arithmetic language will have to only operate on simple
                # numbers.
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                '10': 10,
                '20': 20,
                '-5': -5,
                '-2': -2,
                })

    @predicate
    def add(self, num1: int, num2: int) -> int:
        return num1 + num2

    @predicate
    def sum(self, numbers: List[int]) -> int:
        return sum(numbers)

    # Unfortunately, to make lists, we need to have some function with a fixed number of list
    # elements that we can predict.  No variable number of arguments - that gives us an infinite
    # number of production rules in our grammar.
    @predicate
    def list1(self, num1: int) -> List[int]:
        return [num1]

    @predicate
    def list2(self, num1: int, num2: int) -> List[int]:
        return [num1, num2]

    @predicate
    def list3(self, num1: int, num2: int, num3: int) -> List[int]:
        return [num1, num2, num3]

    @predicate
    def list4(self, num1: int, num2: int, num3: int, num4: int) -> List[int]:
        return [num1, num2, num3, num4]

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
        return num1 // num2

    @predicate
    def halve(self, num1: int) -> int:
        return num1 // 2

    @predicate
    def three(self) -> int:
        return 3

    @predicate
    def three_less(self, function: Callable[[int, int], int]) -> Callable[[int, int], int]:
        """
        Wraps a function into a new function that always returns three less than what the original
        function would.  Totally senseless function that's just here to test higher-order
        functions.
        """
        def new_function(num1: int, num2: int) -> int:
            return function(num1, num2) - 3
        return new_function

    def not_a_predicate(self) -> int:
        return 5


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class DomainLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.language = Arithmetic()

    def test_constant_logical_form(self):
        assert self.language.execute('5') == 5
        assert self.language.execute('2') == 2
        assert self.language.execute('20') == 20
        assert self.language.execute('3') == 3
        with pytest.raises(ExecutionError, match='Unrecognized constant'):
            self.language.execute('"add"')

    def test_error_message_with_wrong_arguments(self):
        with pytest.raises(ExecutionError):
            self.language.execute('(add)')
        with pytest.raises(ExecutionError):
            self.language.execute('(add 2)')

    def test_not_all_functions_are_predicates(self):
        # This should not execute to 5, but instead be treated as a constant.
        with pytest.raises(ExecutionError, match='Unrecognized constant'):
            self.language.execute('not_a_predicate')

    def test_basic_logical_form(self):
        assert self.language.execute('three') == 3
        assert self.language.execute('(add 2 3)') == 5
        assert self.language.execute('(subtract 2 3)') == -1
        assert self.language.execute('(halve 20)') == 10

    def test_list_types(self):
        assert self.language.execute('(sum (list1 2))') == 2
        assert self.language.execute('(sum (list2 2 3))') == 5
        assert self.language.execute('(sum (list4 2 10 -2 -5))') == 5
        assert self.language.execute('(sum (list4 2 three (halve 4) (add -5 -2)))') == 0

    def test_nested_logical_form(self):
        assert self.language.execute('(add 2 (subtract 4 2))') == 4
        assert self.language.execute('(halve (multiply (divide 9 3) (power 2 3)))') == 12

    def test_higher_order_logical_form(self):
        assert self.language.execute('((three_less add) 2 (subtract 4 2))') == 1

    def test_execute_action_sequence(self):
        # Repeats tests from above, but using `execute_action_sequence` instead of `execute`.
        logical_form = '(add 2 (subtract 4 2))'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.execute_action_sequence(action_sequence) == 4
        logical_form = '(halve (multiply (divide 9 3) (power 2 3)))'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.execute_action_sequence(action_sequence) == 12
        logical_form = '((three_less add) 2 (subtract 4 2))'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.execute_action_sequence(action_sequence) == 1
        logical_form = '((three_less add) three (subtract 4 2))'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.execute_action_sequence(action_sequence) == 2

    def test_get_nonterminal_productions(self):
        valid_actions = self.language.get_nonterminal_productions()
        assert set(valid_actions.keys()) == {
                '@start@',
                'int',
                'List[int]',
                '<int:int>',
                '<int,int:int>',
                '<List[int]:int>',
                '<int:List[int]>',
                '<int,int:List[int]>',
                '<int,int,int:List[int]>',
                '<int,int,int,int:List[int]>',
                '<<int,int:int>:<int,int:int>>',
                }
        check_productions_match(valid_actions['@start@'],
                                ['int'])
        check_productions_match(valid_actions['int'],
                                ['[<int,int:int>, int, int]', '[<int:int>, int]',
                                 '[<List[int]:int>, List[int]]', 'three', '1',
                                 '2', '3', '4', '5', '6', '7', '8', '9', '10', '20', '-5', '-2'])
        check_productions_match(valid_actions['List[int]'],
                                ['[<int:List[int]>, int]',
                                 '[<int,int:List[int]>, int, int]',
                                 '[<int,int,int:List[int]>, int, int, int]',
                                 '[<int,int,int,int:List[int]>, int, int, int, int]'])
        check_productions_match(valid_actions['<int:int>'],
                                ['halve'])
        check_productions_match(valid_actions['<int,int:int>'],
                                ['[<<int,int:int>:<int,int:int>>, <int,int:int>]',
                                 'add',
                                 'subtract',
                                 'multiply',
                                 'divide',
                                 'power'])
        check_productions_match(valid_actions['<List[int]:int>'],
                                ['sum'])
        check_productions_match(valid_actions['<int:List[int]>'],
                                ['list1'])
        check_productions_match(valid_actions['<int,int:List[int]>'],
                                ['list2'])
        check_productions_match(valid_actions['<int,int,int:List[int]>'],
                                ['list3'])
        check_productions_match(valid_actions['<int,int,int,int:List[int]>'],
                                ['list4'])
        check_productions_match(valid_actions['<<int,int:int>:<int,int:int>>'],
                                ['three_less'])

    def test_logical_form_to_action_sequence(self):
        action_sequence = self.language.logical_form_to_action_sequence('(add 2 3)')
        assert action_sequence == ['@start@ -> int',
                                   'int -> [<int,int:int>, int, int]',
                                   '<int,int:int> -> add',
                                   'int -> 2',
                                   'int -> 3']

        action_sequence = self.language.logical_form_to_action_sequence('(halve (subtract 8 three))')
        assert action_sequence == ['@start@ -> int',
                                   'int -> [<int:int>, int]',
                                   '<int:int> -> halve',
                                   'int -> [<int,int:int>, int, int]',
                                   '<int,int:int> -> subtract',
                                   'int -> 8',
                                   'int -> three']

        logical_form = '(halve (multiply (divide 9 three) (power 2 3)))'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> int',
                                   'int -> [<int:int>, int]',
                                   '<int:int> -> halve',
                                   'int -> [<int,int:int>, int, int]',
                                   '<int,int:int> -> multiply',
                                   'int -> [<int,int:int>, int, int]',
                                   '<int,int:int> -> divide',
                                   'int -> 9',
                                   'int -> three',
                                   'int -> [<int,int:int>, int, int]',
                                   '<int,int:int> -> power',
                                   'int -> 2',
                                   'int -> 3']

    def test_logical_form_to_action_sequence_with_higher_order_functions(self):
        action_sequence = self.language.logical_form_to_action_sequence('((three_less add) 2 3)')
        assert action_sequence == ['@start@ -> int',
                                   'int -> [<int,int:int>, int, int]',
                                   '<int,int:int> -> [<<int,int:int>:<int,int:int>>, <int,int:int>]',
                                   '<<int,int:int>:<int,int:int>> -> three_less',
                                   '<int,int:int> -> add',
                                   'int -> 2',
                                   'int -> 3']

    def test_action_sequence_to_logical_form(self):
        logical_form = '(add 2 3)'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        recovered_logical_form = self.language.action_sequence_to_logical_form(action_sequence)
        assert recovered_logical_form == logical_form

        logical_form = '(halve (multiply (divide 9 three) (power 2 3)))'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        recovered_logical_form = self.language.action_sequence_to_logical_form(action_sequence)
        assert recovered_logical_form == logical_form

        logical_form = '((three_less add) 2 3)'
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        recovered_logical_form = self.language.action_sequence_to_logical_form(action_sequence)
        assert recovered_logical_form == logical_form

    def test_logical_form_parsing_fails_on_bad_inputs(self):
        # We don't catch all type inconsistencies in the code, but we _do_ catch some.  If we add
        # more that we catch, this is a good place to test for them.
        with pytest.raises(ParsingError, match='Wrong number of arguments'):
            self.language.logical_form_to_action_sequence('(halve 2 3)')
        with pytest.raises(ParsingError, match='Wrong number of arguments'):
            self.language.logical_form_to_action_sequence('(add 3)')
        with pytest.raises(ParsingError, match='unallowed start type'):
            self.language.logical_form_to_action_sequence('add')
        with pytest.raises(ParsingError, match='Zero-arg function or constant'):
            self.language.logical_form_to_action_sequence('(sum (3 2))')
        with pytest.raises(ParsingError, match='did not have expected type'):
            self.language.logical_form_to_action_sequence('(sum (add 2 3))')

    def test_execution_with_side_arguments(self):
        class SideArgumentLanguage(DomainLanguage):
            def __init__(self) -> None:
                super().__init__(start_types={int}, allowed_constants={'1': 1, '2': 2, '3': 3})
            @predicate_with_side_args(['num2'])
            def add(self, num1: int, num2: int) -> int:
                return num1 + num2

            @predicate_with_side_args(['num'])
            def current_number(self, num: int) -> int:
                return num

        language = SideArgumentLanguage()

        # (add 1)
        action_sequence = ['@start@ -> int',
                           'int -> [<int:int>, int]',
                           '<int:int> -> add',
                           'int -> 1']
        # For each action in the action sequence, we pass state.  We only actually _use_ the state
        # when the action we've predicted at that step needs the state.  In this case, the third
        # action will get {'num2': 3} passed to the `add()` function.
        state = [{'num2': 1}, {'num2': 2}, {'num2': 3}, {'num2': 4}]
        assert language.execute_action_sequence(action_sequence, state) == 4

        # (add current_number)
        action_sequence = ['@start@ -> int',
                           'int -> [<int:int>, int]',
                           '<int:int> -> add',
                           'int -> current_number']
        state = [{'num2': 1, 'num': 5},
                 {'num2': 2, 'num': 6},
                 {'num2': 3, 'num': 7},
                 {'num2': 4, 'num': 8}]
        assert language.execute_action_sequence(action_sequence, state) == 11

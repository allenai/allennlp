from typing import Any, Callable, Dict, List, Set, Type
import inspect
import logging
import types
import typing

from nltk.sem.logic import Type as NltkType

from allennlp.semparse import util
from allennlp.semparse.type_declarations import type_declaration
from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType, NameMapper

logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """
    This exception gets raised when there is a parsing error during logical form processing.  This
    might happen because you're not handling the full set of possible logical forms, for instance,
    and having this error provides a consistent way to catch those errors and log how frequently
    this occurs.
    """
    def __init__(self, message):
        super(ParsingError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class ExecutionError(Exception):
    """
    This exception gets raised when you're trying to execute a logical form that your executor does
    not understand. This may be because your logical form contains a function with an invalid name
    or a set of arguments whose types do not match those that the function expects.
    """
    def __init__(self, message):
        super(ExecutionError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def predicate(function: Callable) -> Callable:  # pylint: disable=invalid-name
    function.is_predicate = True
    return function


class Executor:
    def __init__(self):
        self._type_map: Dict[Type, NltkType] = {}
        self._functions: Dict[str, Callable] = {}
        self._function_types: Dict[str, Tuple[List[NltkType], NltkType]] = {}
        self._name_mapper = NameMapper()
        self.start_types: Set[Type] = None
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if hasattr(function, 'is_predicate') and function.is_predicate:
                    self._add_predicate(name, function)
        # Caching this to avoid recompting it every time `get_valid_actions` is called.
        self._valid_actions: Dict[str, List[str]] = None

    def execute(self, logical_form: str):
        logical_form = logical_form.replace(",", " ")
        expression_as_list = util.lisp_to_nested_expression(logical_form)
        return self._handle_expression(expression_as_list)

    def get_valid_actions(self) -> Dict[str, List[str]]:
        if not self._valid_actions:
            basic_types = [nltk_type for nltk_type in self._type_map.values()
                           if isinstance(nltk_type, NamedBasicType)]
            if self.start_types:
                start_types = set([self.get_basic_nltk_type(type_) for type_ in self.start_types])
            else:
                start_types = None
            self._valid_actions = type_declaration.get_valid_actions(self._name_mapper.name_mapping,
                                                                     self._name_mapper.type_signatures,
                                                                     basic_types,
                                                                     valid_starting_types=start_types)
        return self._valid_actions

    def logical_form_to_action_sequence(self, logical_form: str) -> List[str]:
        expression = util.lisp_to_nested_expression(logical_form)
        try:
            transitions = self._get_transitions(expression, expected_type=None)
        except ParsingError:
            logger.error(f'Error parsing logical form: {logical_form}')
            raise
        # TODO(mattg): we _could_ check that the start_type here is one of self.start_types.
        start_type = transitions[0].split(' -> ')[0]
        transitions.insert(0, f'@start@ -> {start_type}')
        return transitions

    def action_sequence_to_logical_form(self, action_sequence: List[str]) -> str:
        pass

    def _add_predicate(self, name: str, function: types.MethodType):
        self._functions[name] = function
        signature = inspect.signature(function)
        argument_types = [param.annotation for param in signature.parameters.values()]
        return_type = signature.return_annotation
        argument_nltk_types = [self.get_basic_nltk_type(arg_type) for arg_type in argument_types]
        return_nltk_type = self.get_basic_nltk_type(return_type)
        function_nltk_type = self.get_function_type(argument_nltk_types, return_nltk_type)
        self._function_types[name] = (argument_nltk_types, return_nltk_type)
        self._name_mapper.map_name_with_signature(name, function_nltk_type)

    def get_basic_nltk_type(self, type_: Type) -> NltkType:
        if type_ not in self._type_map:
            if isinstance(type_, typing.GenericMeta):
                # This is something like List[int].  type_.__name__ will only give 'List', though, so
                # we need to do some magic here.
                origin = type_.__origin__
                args = type_.__args__
                name = '_'.join([origin.__name__] + [arg.__name__ for arg in args])
                # TODO(mattg): we probably need to add a grammar rule in here too, somehow, so we
                # can actually _produce_ a list of things.
            else:
                name = type_.__name__
            self._type_map[type_] = NamedBasicType(name.upper())
        return self._type_map[type_]

    def get_function_type(self, arg_types: List[NltkType], return_type: NltkType) -> NltkType:
        if not arg_types:
            # Functions with no arguments are basically constants whose type match their return
            # type.
            return return_type

        # NLTK's logic classes _curry_ multi-argument functions.  A `ComplexType` is a
        # single-argument function, and you nest these for multi-argument functions.  This for loop
        # does the nesting for multi-argument functions.
        right_argument = return_type
        for arg_type in reversed(arg_types):
            final_type = ComplexType(arg_type, right_argument)
            right_argument = final_type
        return final_type

    def _handle_expression(self, expression: Any):
        if isinstance(expression, (list, tuple)):
            if expression[0] in self._functions:
                function = self._functions[expression[0]]
                arguments = [self._handle_expression(arg) for arg in expression[1:]]
                try:
                    return function(*arguments)
                except (TypeError, ValueError) as error:
                    raise ExecutionError("Error executing expression {expression}: {error}")
            else:
                return [self._handle_expression(item) for item in expression]
        elif isinstance(expression, str):
            if expression[0] == '"' and expression[-1] == '"':
                return expression[1:-1]
            if expression[0] == "'" and expression[-1] == "'":
                return expression[1:-1]
            if expression in self._functions:
                try:
                    return self._functions[expression]()
                except (TypeError, ValueError):
                    raise ExecutionError("Error executing expression {expression}: {error}")
            try:
                int_value = int(expression)
                return int_value
            except ValueError:
                pass
            try:
                float_value = float(expression)
                return float_value
            except ValueError:
                pass
        return expression

    def _get_transitions(self, expression: Any, expected_type: NltkType) -> List[str]:
        if isinstance(expression, (list, tuple)):
            if expression[0] in self._functions:
                name = expression[0]
                argument_types, return_type = self._function_types[name]
                function_type = self._name_mapper.get_signature(name)
                if expected_type and expected_type != return_type:
                    raise ParsingError(f'{expression} did not have expected type {expected_type}')
                right_side = f'[{function_type}, {", ".join(str(arg) for arg in argument_types)}]'
                first_transition = f'{return_type} -> {right_side}'
                second_transition = f'{function_type} -> {name}'
                argument_transitions = []
                if len(argument_types) != len(expression[1:]):
                    raise ParsingError(f'Wrong number of arguments for function in {expression}')
                for argument_type, subexpression in zip(argument_types, expression[1:]):
                    argument_transitions.extend(self._get_transitions(subexpression, argument_type))
                return [first_transition, second_transition] + argument_transitions
            else:
                raise ParsingError('Bare lists not implemented yet')
        elif isinstance(expression, str):
            # TODO(mattg): do type inference on constants, and check for matches (and allow
            # top-level constant expressions).
            # TODO(mattg): this could also possibly cause problems with inconsistent handling of
            # quotation marks.  We should probably pull the constant handling out into its own
            # function in both `_get_transitions` and `_handle_expression`.
            if expected_type:
                return [f'{expected_type} -> {expression}']
            else:
                raise ParsingError('Constant expressions not implemented yet')
        else:
            raise ParsingError('Not sure how you got here.  Raise an issue on github with details')

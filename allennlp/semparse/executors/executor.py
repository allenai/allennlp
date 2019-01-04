from typing import Any, Callable, Dict, List, Set, Type
import inspect
import types
import typing

from nltk.sem.logic import Type as NltkType

from allennlp.semparse import util
from allennlp.semparse.type_declarations.type_declaration import (ComplexType, NamedBasicType,
                                                                  NameMapper, get_valid_actions)


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
        self._functions = {}
        self._name_mapper = NameMapper()
        self.start_types: Set[Type] = None
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if hasattr(function, 'is_predicate') and function.is_predicate:
                    self._add_predicate(name, function)

    def execute(self, logical_form: str):
        logical_form = logical_form.replace(",", " ")
        expression_as_list = util.lisp_to_nested_expression(logical_form)
        return self._handle_expression(expression_as_list)

    def get_valid_actions(self) -> Dict[str, List[str]]:
        basic_types = [nltk_type for nltk_type in self._type_map.values()
                       if isinstance(nltk_type, NamedBasicType)]
        if self.start_types:
            start_types = set([self.get_basic_nltk_type(type_) for type_ in self.start_types])
        else:
            start_types = None
        return get_valid_actions(self._name_mapper.common_name_mapping,
                                 self._name_mapper.common_type_signature,
                                 basic_types,
                                 valid_starting_types=start_types)

    def _add_predicate(self, name: str, function: types.MethodType):
        self._functions[name] = function
        signature = inspect.signature(function)
        argument_types = [param.annotation for param in signature.parameters.values()]
        return_type = signature.return_annotation
        argument_nltk_types = [self.get_basic_nltk_type(arg_type) for arg_type in argument_types]
        return_nltk_type = self.get_basic_nltk_type(return_type)
        function_nltk_type = self.get_function_type(argument_nltk_types, return_nltk_type)
        self._name_mapper.map_name_with_signature(name, function_nltk_type)
        # TODO(mattg): figure out what to do about the `curried_functions` stuff.

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

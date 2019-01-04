from typing import Any, Callable, Dict, Type
import types

from allennlp.semparse import util


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
        self._functions = {}
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if hasattr(function, 'is_predicate') and function.is_predicate:
                    self._add_predicate(name, function)

    def _add_predicate(self, name: str, function: types.MethodType):
        self._functions[name] = function
        # TODO(matt): inspect the type annotations of the function, add function to grammar.

    def execute(self, logical_form: str):
        logical_form = logical_form.replace(",", " ")
        expression_as_list = util.lisp_to_nested_expression(logical_form)
        return self._handle_expression(expression_as_list)

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

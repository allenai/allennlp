from typing import Union, List, Any

from allennlp.semparse import util as semparse_util

NestedList = List[Union[str, List]]  # pylint: disable=invalid-name

class Executor:
    """
    Abstract class that executors for various domains can inherit from. They need to override
    the `_handle_expression` method.

    Parameters
    ----------
    context : ``Any``
        The context relevant for execution, if any. Objects of this class are deemed equal if these
        fields of the objects are equal.
    """
    def __init__(self, context: Any) -> None:
        self.context = context

    def execute(self, logical_form: str) -> Any:
        if not logical_form.startswith("("):
            logical_form = f"({logical_form})"
        logical_form = logical_form.replace(",", " ")
        expression_as_list = semparse_util.lisp_to_nested_expression(logical_form)
        # Expression list has an additional level of nesting at the top. For example, if the logical
        # for is "(select all_rows fb:row.row.league)", the expression list will be
        # [['select', 'all_rows', 'fb:row.row.league']].
        # Removing the top most level of nesting.
        result = self._handle_expression(expression_as_list[0])
        return result

    def _handle_expression(self, expression_list: NestedList) -> Any:
        raise NotImplementedError

    def __eq__(self, other):
        return self.context == other.context

from typing import List, Dict, Union
import pyparsing

from nltk.sem.logic import Expression, LambdaExpression, Type

from allennlp.data.semparse.type_declarations.type_declaration import DynamicTypeLogicParser


class World:
    """
    Base class for defining a world in a new domain. This class defines a method to translate a logical form
    as per a naming convention that works with NLTK's ``LogicParser``. The sub-classes can decide on the
    convention by overriding the ``_map_name`` method that does token level mapping. This class also defines
    a method for transforming ``Expressions`` into action sequences.
    """
    def __init__(self,
                 constant_type_prefixes: Dict[str, str] = None,
                 global_type_signatures: Dict[str, Type] = None):
        # NLTK has a naming convention for variable types. If the world has predicate or entity names beyond
        # what's defined in the COMMON_NAME_MAPPING, they need to be added to this dict.
        # We initialize this dict with common predicate names and update it as we process logical forms.
        self.local_name_mapping = {}
        # Similarly, these are the type signatures not in the COMMON_TYPE_SIGNATURE.
        self.local_type_signatures = {}
        type_prefixes = constant_type_prefixes or {}
        self.global_type_signatures = global_type_signatures or {}
        self._logic_parser = DynamicTypeLogicParser(constant_type_prefixes=type_prefixes,
                                                    type_signatures=self.global_type_signatures)

    def parse_logical_form(self, logical_form: str) -> Expression:
        def _process_nested_expression(nested_expression: List[Union[str, List[str]]]) -> str:
            """
            ``nested_expression`` is the result of parsing a Lambda-DCS expression in Lisp format.
            We process it recursively and return a string in the format that NLTK's ``LogicParser``
            would understand.
            """
            expression_is_list = isinstance(nested_expression, list)
            expression_size = len(nested_expression)
            if expression_is_list and expression_size == 1 and isinstance(nested_expression[0], list):
                return _process_nested_expression(nested_expression[0])
            elements_are_leaves = [isinstance(element, str) for element in nested_expression]
            if all(elements_are_leaves):
                mapped_names = [self._map_name(name) for name in nested_expression]
            else:
                mapped_names = []
                for element, is_leaf in zip(nested_expression, elements_are_leaves):
                    if is_leaf:
                        mapped_names.append(self._map_name(element))
                    else:
                        mapped_names.append(_process_nested_expression(element))
            if mapped_names[0] == "\\":
                # This means the predicate is lambda. NLTK wants the variable name to not be within parantheses.
                # Adding parentheses after the variable.
                arguments = [mapped_names[1]] + ["(%s)" % name for name in mapped_names[2:]]
            else:
                arguments = ["(%s)" % name for name in mapped_names[1:]]
            return "(%s %s)" % (mapped_names[0], " ".join(arguments))
        parsed_lisp = pyparsing.OneOrMore(pyparsing.nestedExpr()).parseString(logical_form).asList()
        translated_string = _process_nested_expression(parsed_lisp)
        type_signature = self.local_type_signatures.copy()
        type_signature.update(self.global_type_signatures)
        return self._logic_parser.parse(translated_string, signature=type_signature)

    def _map_name(self, name: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_action_sequence(expression: Expression) -> List[str]:
        """
        Returns the sequence of actions (as strings) that resulted in the given expression.
        """
        def _get_transitions(expression: Expression,
                             current_transitions: List[str]) -> List[str]:
            expression_type = expression.type
            try:
                # ``Expression.visit()`` takes two arguments: the first one is a function applied on each
                # sub-expression and the second is a combinator that is applied to the list of values returned
                # from the function applications. We just want the list of all sub-expressions here.
                sub_expressions = expression.visit(lambda x: x, lambda x: x)
                transformed_types = [sub_exp.type for sub_exp in sub_expressions]
                if isinstance(expression, LambdaExpression):
                    # If the expression is a lambda expression, the list of sub expressions does not include
                    # the "lambda x" term. We're adding it here so that we will see transitions like
                    #   <e,d> -> [\x, d] instead of
                    #   <e,d> -> [d]
                    transformed_types = ["/X"] + transformed_types
                current_transitions.append("%s -> %s" % (expression_type,
                                                         str(transformed_types)))
                for sub_expression in sub_expressions:
                    _get_transitions(sub_expression, current_transitions)
            except NotImplementedError:
                # This means that the expression is a leaf. We simply make a transition from its type to itself.
                current_transitions.append("%s -> %s" % (expression_type, expression))
            return current_transitions
        # Starting with the type of the whole expression
        return _get_transitions(expression, [str(expression.type)])

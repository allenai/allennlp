import re

from typing import List
import pyparsing

from nltk.sem.logic import Expression, LambdaExpression

from allennlp.data.dataset_readers.wikitables.type_declaration import COLUMN_TYPE, CELL_TYPE, PART_TYPE
from allennlp.data.dataset_readers.wikitables.type_declaration import COMMON_NAME_MAPPING, COMMON_TYPE_SIGNATURE

from allennlp.data.dataset_readers.wikitables.type_declaration import DynamicTypeLogicParser
from allennlp.data.dataset_readers.wikitables.table import TableKnowledgeGraph


class World:
    """
    We store all the information related to a world (i.e. the context in which logical forms will be
    executed) here. For WikiTableQuestions, this includes a representation of a table, mapping from
    Sempre variables in all logical forms to NLTK variables, and the types of all predicates and entities.
    """
    def __init__(self, table_graph: TableKnowledgeGraph) -> None:
        self.table_graph = table_graph
        # NLTK has a naming convention for variable types (see ``_map_name`` for more details).
        # We initialize this dict with common predicate names and update it as we process logical forms.
        # TODO (pradeep): Should we do updates while reading tables instead?
        self.name_mapping = COMMON_NAME_MAPPING.copy()
        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0
        # Initial type signature. Will update when we see more predicates.
        self.type_signature = COMMON_TYPE_SIGNATURE.copy()
        self._logic_parser = DynamicTypeLogicParser(type_check=True)

    def process_sempre_forms(self, sempre_forms: List[str]) -> List[Expression]:
        """
        Processes logical forms in Sempre format (that are either gold annotations or output from DPD)
        and converts them into ``Expression`` objects in NLTK. The conversion involves mapping names to
        follow NLTK's variable naming conventions.
        """
        expressions = []
        for sempre_form in sempre_forms:
            parsed_lisp = pyparsing.OneOrMore(pyparsing.nestedExpr()).parseString(sempre_form).asList()
            translated_string = self._process_nested_expression(parsed_lisp)
            expression = self._logic_parser.parse(translated_string, signature=self.type_signature)
            expressions.append(expression)
        return expressions

    def _process_nested_expression(self, nested_expression):
        """
        ``nested_expression`` is the result of parsing a Lambda-DCS expression in Lisp format. We process it
        recursively and return a string in the format that NLTK's ``LogicParser`` would understand.
        """
        expression_is_list = isinstance(nested_expression, list)
        expression_size = len(nested_expression)
        if expression_is_list and expression_size == 1 and isinstance(nested_expression[0], list):
            return self._process_nested_expression(nested_expression[0])
        elements_are_leaves = [isinstance(element, str) for element in nested_expression]
        if all(elements_are_leaves):
            mapped_names = [self._map_name(name) for name in nested_expression]
        else:
            mapped_names = []
            for element, is_leaf in zip(nested_expression, elements_are_leaves):
                if is_leaf:
                    mapped_names.append(self._map_name(element))
                else:
                    mapped_names.append(self._process_nested_expression(element))
        if mapped_names[0] == "\\":
            # This means the predicate is lambda. NLTK wants the variable name to not be within parantheses.
            arguments = [mapped_names[1]] + ["(%s)" % name for name in mapped_names[2:]]
        else:
            arguments = ["(%s)" % name for name in mapped_names[1:]]
        return "(%s %s)" % (mapped_names[0], " ".join(arguments))

    def _map_name(self, sempre_name: str) -> str:
        """
        Takes the name of a predicate or a constant as used by Sempre, maps it to a unique string such that
        NLTK processes it appropriately. This is needed because NLTK has a naming convention for variables:
            Function variables: Single upper case letter optionally foolowed by digits
            Individual variables: Single lower case letter (except e for events) optionally followed by digits
            Constants: Everything else

        Parameters
        ----------
        sempre_name : str
            Token from Sempre's logical form.
        """
        if sempre_name not in self.name_mapping:
            if sempre_name.startswith("fb:row.row"):
                # Column name
                translated_name = "C%d" % self._column_counter
                self._column_counter += 1
                self.type_signature[translated_name] = COLUMN_TYPE
                self.name_mapping[sempre_name] = translated_name
            elif sempre_name.startswith("fb:cell"):
                # Cell name
                translated_name = "cell:%s" % sempre_name.split(".")[-1]
                self.type_signature[translated_name] = CELL_TYPE
                self.name_mapping[sempre_name] = translated_name
            elif sempre_name.startswith("fb:part"):
                # part name
                translated_name = "part:%s" % sempre_name.split(".")[-1]
                self.type_signature[translated_name] = PART_TYPE
                self.name_mapping[sempre_name] = translated_name
            else:
                # NLTK throws an error if it sees a "." in constants, which will most likely happen within
                # numbers as a decimal point. We're changing those to underscores.
                translated_name = sempre_name.replace(".", "_")
                if re.match("-[0-9_]+", translated_name):
                    # The string is a negative number. This makes NLTK interpret this as a negated expression
                    # and force its type to be TRUTH_VALUE (t).
                    translated_name = translated_name.replace("-", "~")
        else:
            translated_name = self.name_mapping[sempre_name]
        return translated_name

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
        return _get_transitions(expression, [])

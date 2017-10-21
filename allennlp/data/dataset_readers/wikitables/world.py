from typing import List
from pyparsing import OneOrMore, nestedExpr

from nltk.sem.logic import LogicParser, Expression

from allennlp.data.dataset_readers.wikitables import WTD, TableKnowledgeGraph


class World:
    """
    We store all the information related to a world (i.e. the context in which logical forms will be
    executed) here. For WikiTableQuestions, this includes a representation of a table, mapping from
    Sempre variables in all logical forms to NLTK variables, and the types of all predicates and entities.
    """
    def __init__(self, table_filename: str) -> None:
        self.table_graph = TableKnowledgeGraph.read_table_from_tsv(table_filename)
        # NLTK has a naming convention for variable types (see ``_map_name`` for more details).
        # We initialize this dict with common predicate names and update it as we process logical forms.
        # TODO: Should we do updates while reading tables instead?
        self.name_mapping = {"reverse": "R", "max": "M0", "min": "M1",
                             "argmax": "A0", "argmin": "A1",
                             "and": "A", "or": "O", "next": "N",
                             "fb:cell.cell.date": "D", "fb:cell.cell.number": "B"}
        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0
        # Initial type signature. Will update when we see more predicates.
        self.type_signature = {"R": WTD.REVERSE_TYPE, "M0": WTD.MAX_MIN_TYPE, "M1": WTD.MAX_MIN_TYPE,
                               "A0": WTD.MAX_MIN_TYPE, "A1": WTD.MAX_MIN_TYPE}
        self._logic_parser = LogicParser(type_check=True)

    def process_sempre_forms(self, sempre_forms: List[str]) -> List[Expression]:
        """
        Processes logical forms in Sempre format (that are either gold annotations or output from DPD)
        and converts them into ``Expression`` objects in NLTK. The conversion involves mapping names to
        follow NLTK's variable naming conventions.
        """
        expressions = []
        for sempre_form in sempre_forms:
            parsed_lisp = OneOrMore(nestedExpr()).parseString(sempre_form).asList()
            translated_string = self._process_nested_expression(parsed_lisp)
            expression = self._logic_parser.parse(translated_string, signature=self.type_signature)
            expressions.append(expression)
        return expressions

    def _process_nested_expression(self, nested_expression):
        """
        ``nested_expression`` is the result of parsing a Lambda-DCS expression in Lisp format. We process it
        recursively and return a string in the format that NLTK's ``LogicParser`` would understand.
        """
        if isinstance(nested_expression, list) and len(nested_expression) == 1:
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
        return "(%s %s)" % (mapped_names[0], " ".join("(%s)" % name for name in mapped_names[1:]))

    def _map_name(self, sempre_name: str):
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
                self.type_signature[translated_name] = WTD.COLUMN_TYPE
            elif sempre_name.startswith("fb:cell"):
                # Cell name.
                translated_name = "cell:%s" % sempre_name.split(".")[-1]
                self.type_signature[translated_name] = WTD.CELL_TYPE
            self.name_mapping[sempre_name] = translated_name
        return self.name_mapping[sempre_name]

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
                current_transitions.append("%s -> %s" % (expression_type,
                                                         str([sub_exp.type for sub_exp in sub_expressions])))
                for sub_expression in sub_expressions:
                    _get_transitions(sub_expression, current_transitions)
            except NotImplementedError:
                # This means that the expression is a leaf. We simply make a transition from its type to itself.
                current_transitions.append("%s -> %s" % (expression_type, expression))
            return current_transitions
        return _get_transitions(expression, [])

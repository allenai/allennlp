"""
We store all the information related to a world (i.e. the context in which logical forms will be
executed) here. For WikiTableQuestions, this includes a representation of a table, mapping from
Sempre variables in all logical forms to NLTK variables, and the types of all predicates and entities.
"""
from typing import List, Set
import re

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.data.tokenizers import Token
from allennlp.data.semparse.worlds.world import World
from allennlp.data.semparse.type_declarations import type_declaration as base_types
from allennlp.data.semparse.type_declarations import wikitables_type_declaration as types
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph


class WikiTablesWorld(World):
    """
    World representation for the WikitableQuestions domain.

    Parameters
    ----------
    table_graph : ``TableKnowledgeGraph``
        Context associated with this world.
    question_tokens : ``List[Token]``
        The tokenized question, which we use to augment our parser output space.  In particular,
        because there are an infinite number of numbers that we can output, we restrict the space
        of numbers that we consider to just the numbers that appear in the question, plus a few
        small numbers.
    """
    def __init__(self, table_graph: TableKnowledgeGraph, question_tokens: List[Token]) -> None:
        super(WikiTablesWorld, self).__init__(constant_type_prefixes={"part": types.PART_TYPE,
                                                                      "cell": types.CELL_TYPE},
                                              global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                              global_name_mapping=types.COMMON_NAME_MAPPING,
                                              num_nested_lambdas=1)
        self.table_graph = table_graph

        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

        # This adds all of the cell and column names to our local name mapping, so we can get them
        # as valid actions in the parser.
        for entity in table_graph.get_all_entities():
            self._map_name(entity)

        numbers = self._get_numbers_from_tokens(question_tokens) + list(str(i) for i in range(10))
        for number in numbers:
            self._add_name_mapping(number, number, types.CELL_TYPE)

    def _get_numbers_from_tokens(self, tokens: List[Token]) -> List[str]:
        """
        Finds numbers in the input tokens and returns them as strings.

        Eventually, we'd want this to detect things like ordinals ("first", "third") and cardinals
        ("one", "two"), but for now we just look for literal digits, and make up for missing
        ordinals and cardinals by adding all single-digit numbers as possible numbers to output.
        """
        numbers = []
        for token in tokens:
            # We'll use a check for float(text) to find numbers, because text.isdigit() doesn't
            # catch things like "-3" or "0.07".
            try:
                float(token.text)
                numbers.append(token)
            except ValueError:
                pass
        return numbers

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def _map_name(self, name: str) -> str:
        """
        Takes the name of a predicate or a constant as used by Sempre, maps it to a unique string such that
        NLTK processes it appropriately. This is needed because NLTK has a naming convention for variables:
            Function variables: Single upper case letter optionally followed by digits
            Individual variables: Single lower case letter (except e for events) optionally followed by digits
            Constants: Everything else

        Parameters
        ----------
        name : str
            Token from Sempre's logical form.
        """
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if name.startswith("fb:row.row"):
                # Column name
                translated_name = "C%d" % self._column_counter
                self._column_counter += 1
                self._add_name_mapping(name, translated_name, types.COLUMN_TYPE)
            elif name.startswith("fb:cell"):
                # Cell name
                translated_name = "cell:%s" % name.split(".")[-1]
                self._add_name_mapping(name, translated_name, types.CELL_TYPE)
            elif name.startswith("fb:part"):
                # part name
                translated_name = "part:%s" % name.split(".")[-1]
                self._add_name_mapping(name, translated_name, types.PART_TYPE)
            else:
                # NLTK throws an error if it sees a "." in constants, which will most likely happen within
                # numbers as a decimal point. We're changing those to underscores.
                translated_name = name.replace(".", "_")
                if re.match("-[0-9_]+", translated_name):
                    # The string is a negative number. This makes NLTK interpret this as a negated expression
                    # and force its type to be TRUTH_VALUE (t).
                    translated_name = translated_name.replace("-", "~")
                self._add_name_mapping(name, translated_name)
        else:
            if name in types.COMMON_NAME_MAPPING:
                translated_name = types.COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

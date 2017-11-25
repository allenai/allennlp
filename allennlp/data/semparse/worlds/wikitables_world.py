"""
We store all the information related to a world (i.e. the context in which logical forms will be
executed) here. For WikiTableQuestions, this includes a representation of a table, mapping from
Sempre variables in all logical forms to NLTK variables, and the types of all predicates and entities.
"""
import re
from typing import List
from overrides import overrides

from nltk.sem.logic import Expression

from allennlp.data.semparse.worlds.world import World
from allennlp.data.semparse.type_declarations.wikitables_type_declaration import (COMMON_NAME_MAPPING,
                                                                                  COMMON_TYPE_SIGNATURE)
from allennlp.data.semparse.type_declarations import wikitables_type_declaration as types
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph


class WikiTablesWorld(World):
    """
    World representation for the WikitableQuestions domain.

    Parameters
    ----------
    table_graph : ``TableKnowledgeGraph``
        Context associated with this world.
    """
    def __init__(self, table_graph: TableKnowledgeGraph) -> None:
        super(WikiTablesWorld, self).__init__(constant_type_prefixes={"part": types.PART_TYPE,
                                                                      "cell": types.CELL_TYPE},
                                              global_type_signatures=COMMON_TYPE_SIGNATURE)
        self.table_graph = table_graph
        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

    def process_sempre_forms(self, sempre_forms: List[str]) -> List[Expression]:
        """
        Processes logical forms in Sempre format (that are either gold annotations or output from DPD)
        and converts them into ``Expression`` objects in NLTK. The conversion involves mapping names to
        follow NLTK's variable naming conventions.
        """
        expressions = []
        for sempre_form in sempre_forms:
            expressions.append(self.parse_logical_form(sempre_form))
        return expressions

    @overrides
    def _map_name(self, name: str) -> str:
        """
        Takes the name of a predicate or a constant as used by Sempre, maps it to a unique string such that
        NLTK processes it appropriately. This is needed because NLTK has a naming convention for variables:
            Function variables: Single upper case letter optionally foolowed by digits
            Individual variables: Single lower case letter (except e for events) optionally followed by digits
            Constants: Everything else

        Parameters
        ----------
        name : str
            Token from Sempre's logical form.
        """
        if name not in COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if name.startswith("fb:row.row"):
                # Column name
                translated_name = "C%d" % self._column_counter
                self._column_counter += 1
                self.local_type_signatures[translated_name] = types.COLUMN_TYPE
                self.local_name_mapping[name] = translated_name
            elif name.startswith("fb:cell"):
                # Cell name
                translated_name = "cell:%s" % name.split(".")[-1]
                self.local_type_signatures[translated_name] = types.CELL_TYPE
                self.local_name_mapping[name] = translated_name
            elif name.startswith("fb:part"):
                # part name
                translated_name = "part:%s" % name.split(".")[-1]
                self.local_type_signatures[translated_name] = types.PART_TYPE
                self.local_name_mapping[name] = translated_name
            else:
                # NLTK throws an error if it sees a "." in constants, which will most likely happen within
                # numbers as a decimal point. We're changing those to underscores.
                translated_name = name.replace(".", "_")
                if re.match("-[0-9_]+", translated_name):
                    # The string is a negative number. This makes NLTK interpret this as a negated expression
                    # and force its type to be TRUTH_VALUE (t).
                    translated_name = translated_name.replace("-", "~")
        else:
            if name in COMMON_NAME_MAPPING:
                translated_name = COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

"""
This module defines QuarelWorld, with a simple domain theory for reasoning about
qualitative relations.
"""
from typing import List, Dict, Set
import re

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse import util as semparse_util
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from allennlp.semparse.type_declarations.quarel_type_declaration import QuarelTypeDeclaration
from allennlp.semparse.worlds.world import World

class QuarelWorld(World):
    """
    Class defining the QuaRel domain theory world.

    Parameters
    ----------
    """
    def __init__(self,
                 table_graph: KnowledgeGraph,
                 syntax: str,
                 qr_coeff_sets: List[Dict[str, int]] = None) -> None:

        self._syntax = syntax
        self.types = QuarelTypeDeclaration(syntax)
        super().__init__(
                global_type_signatures=self.types.name_mapper.type_signatures,
                global_name_mapping=self.types.name_mapper.name_mapping)
        self.table_graph = table_graph

        # Keep map and counter for each entity type encountered (first letter in entity string)
        self._entity_index_maps: Dict[str, Dict[str, int]] = dict()
        self._entity_counters: Dict[str, int] = dict()

        for entity in table_graph.entities:
            self._map_name(entity, keep_mapping=True)

        self._entity_set = set(table_graph.entities)

        self.qr_coeff_sets = qr_coeff_sets
        if qr_coeff_sets is None:
            if "_friction" in syntax:
                self.qr_coeff_sets = [self.qr_coeff_sets_default[0]]
            else:
                self.qr_coeff_sets = self.qr_coeff_sets_default

    def is_table_entity(self, entity_name: str) -> bool:
        """
        Returns ``True`` if the given entity is one of the entities in the table.
        """
        return entity_name in self._entity_set

    # Keep track of entity counters of various entity types
    def _entity_index(self, entity) -> int:
        entity_type = entity[0]
        if entity_type not in self._entity_counters:
            self._entity_counters[entity_type] = 0
            self._entity_index_maps[entity_type] = dict()
        entity_index_map = self._entity_index_maps[entity_type]
        if entity not in entity_index_map:
            entity_index_map[entity] = self._entity_counters[entity_type]
            self._entity_counters[entity_type] += 1
        return entity_index_map[entity]

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        translated_name = name
        if name in self.types.name_mapper.name_mapping:
            translated_name = self.types.name_mapper.name_mapping[name]
        elif name in self.local_name_mapping:
            translated_name = self.local_name_mapping[name]
        elif name.startswith("a:"):
            translated_name = "A"+str(10+self._entity_index(name))
            self._add_name_mapping(name, translated_name, self.types.attr_function_type)

        return translated_name

    def _get_curried_functions(self) -> Dict[Type, int]:
        return self.types.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return self.types.basic_types

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return self.types.starting_types

    # Simple table for how attributes relates to each other
    # First entry is by convention (above in __init__) the friction subset
    qr_coeff_sets_default = [
            {"friction": 1, "speed": -1, "smoothness": -1, "distance": -1, "heat": 1},
            {"speed": 1, "time": -1},
            {"speed": 1, "distance": 1},
            {"time": 1, "distance": 1},
            {"weight": 1, "acceleration": -1},
            {"strength": 1, "distance": 1},
            {"strength": 1, "thickness": 1},
            {"mass": 1, "gravity": 1},
            {"flexibility": 1, "breakability": -1},
            {"distance": 1, "loudness": -1, "brightness": -1, "apparentSize": -1},
            {"exerciseIntensity": 1, "amountSweat": 1}
    ]

    # Size translation for absolute and relative values
    qr_size = {
            'higher': 1,
            'high': 1,
            'low': -1,
            'lower': -1
    }

    def _get_qr_coeff(self, attr1, attr2):
        for qset in self.qr_coeff_sets:
            if attr1 in qset and attr2 in qset:
                return qset[attr1] * qset[attr2]
        return 0

    def _check_compatible(self, setup: List, answer: List) -> bool:
        attributes = {setup[0], answer[0]}
        qr_coeff = None
        for qr_coeff_set in self.qr_coeff_sets:
            if not attributes - qr_coeff_set.keys():
                qr_coeff = qr_coeff_set
        if qr_coeff is None:
            return False  # No compatible attribute sets found

        attribute_dir = qr_coeff[setup[0]] * qr_coeff[answer[0]]
        change_same = 1 if self.qr_size[setup[1]] == self.qr_size[answer[1]] else -1
        world_same = 1 if setup[2] == answer[2] else -1
        return attribute_dir * change_same * world_same == 1

    def _exec_infer(self, setup, *answers):
        answer_index = -1
        if len(answers) == 1:
            if self._check_compatible(setup, answers[0]):
                return 1
            else:
                return 0
        for index, answer in enumerate(answers):
            if self._check_compatible(setup, answer):
                if answer_index > -1:
                    # Found two valid answers
                    answer_index = -2
                else:
                    answer_index = index
        return answer_index

    def _exec_and(self, expr):
        if not expr or expr[0] != 'and':
            return expr
        args = expr[1:]
        if len(args) == 1:
            return args[0]
        if len(args) > 2:
            # More than 2 arguments not allowed by current grammar
            return None
        if self._check_compatible(args[0], args[1]):
            # Check that arguments are compatible, then fine to keep just one
            return args[0]
        return None

    def execute(self, lf_raw: str) -> int:
        """
        Very basic model for executing friction logical forms. For now returns answer index (or
        -1 if no answer can be concluded)
        """
        # Remove "a:" prefixes from attributes (hack)
        logical_form = re.sub(r"\(a:", r"(", lf_raw)
        parse = semparse_util.lisp_to_nested_expression(logical_form)
        if len(parse) < 2:
            return -1
        if parse[0] == 'infer':
            args = [self._exec_and(arg) for arg in parse[1:]]
            if None in args:
                return -1
            return self._exec_infer(*args)
        return -1

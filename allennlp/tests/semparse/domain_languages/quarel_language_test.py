from typing import List, Callable, Tuple

import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import DomainLanguage, ExecutionError, ParsingError, predicate

from pprint import pprint

class Property():
    def __init__(self, name):
        self.name = name 

class World():
    def __init__(self, number):
        self.number = number

class Direction():
    def __init__(self, number):
        self.number = number 

class QuaRelType():
    def __init__(self, prop: Property, direction, world):
        self.prop = prop 
        self.direction = direction
        self.world = world

class QuaRel(DomainLanguage):
    def __init__(self):
        super().__init__(start_types={int}, allowed_constants={'world1': World(1),
                                                               'world2': World(2),
                                                               'higher': Direction(1),
                                                               'lower': Direction(-1),
                                                               'high': Direction(1),
                                                               'low': Direction(-1)})
        self.default_theories = [
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
            {"exerciseIntensity": 1, "amountSweat": 1}]

        for prop in ["friction", "speed", "distance", "heat", "smoothness", "acceleration", "amountSweat", "apparentSize", "breakability",
                     "brightness", "exerciseIntensity", "flexibility", "gravity", "loudness", "mass", "strength", "thickness", "time"
                     "weight"]:
            func = self.make_property_predicate(prop)
            self.add_predicate(prop, func)
        
        # ``and`` is a reserved word, so we add it as a predicate here instead of using the decorator. 
        def and_function(quaval_1: QuaRelType, quaval_2: QuaRelType) -> QuaRelType:
            if self._check_compatible(quaval_1, quaval_2):
                return quaval_1
            else:
                return None
        self.add_predicate('and', and_function)


    def make_property_predicate(self, property_name: str) -> Callable[[Direction, World], QuaRelType]:
        def property_function(direction: Direction, world: World) -> QuaRelType:
            return QuaRelType(Property(property_name), direction, world)
        return property_function

    def _check_compatible(self, quarel_0: QuaRelType, quarel_1: QuaRelType) -> bool:
        for theory in self.default_theories:
            if quarel_0.prop.name in theory and quarel_1.prop.name in theory:
                world_same = 1 if quarel_0.world.number == quarel_1.world.number else -1
                direction_same = 1 if quarel_0.direction.number == quarel_1.direction.number else -1
                if theory[quarel_0.prop.name] * theory[quarel_1.prop.name] * world_same * direction_same == 1:
                    return True
                else:
                    return False
    
    @predicate
    def infer(self, question: QuaRelType, answer_0: QuaRelType, answer_1: QuaRelType) -> int:
        if self._check_compatible(question, answer_0):
            if self._check_compatible(question, answer_1):
                # Found two answers
                return -2
            else:
                return 0
        elif self._check_compatible(question, answer_1):
            return 1
        else:
            return -1


class QuaRelLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.language = QuaRel()
    
    def test_constant_logical_form(self):
        print('execute', self.language.execute('world1'))
            
    def test_infer_quarel(self):
        assert self.language.execute('(infer (speed higher world1) (friction higher world1) (friction lower world1))') == 1
        print(self.language.logical_form_to_action_sequence('(infer (speed higher world1) (friction higher world1) (friction lower world1))'))

        assert self.language.execute('(infer (speed higher world2) (friction higher world1) (friction lower world1))') == 0

    def test_infer_quaval(self):
        assert self.language.execute('(infer (and (thickness low world1) (thickness high world2)) (strength lower world1) (strength lower world2))') == 0
        print(self.language.logical_form_to_action_sequence('(infer (and (thickness low world1) (thickness high world2)) (strength lower world1) (strength lower world2))'))







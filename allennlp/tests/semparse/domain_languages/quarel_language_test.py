from typing import List, Callable, Tuple

import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import DomainLanguage, ExecutionError, ParsingError, predicate

from pprint import pprint

class Number():
    def __init__(self, name):
        self.name = name 

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
        super().__init__(start_types={Number}, allowed_constants={'world1': World(1),
                                                                  'world2': World(2),
                                                                  'higher': Direction(1),
                                                                  'lower': Direction(-1)})
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


    def make_property_predicate(self, property_name: str) -> Callable[[Direction, World], QuaRelType]:
        def property_function(direction: Direction, world: World) -> QuaRelType:
            return QuaRelType(Property(property_name), direction, world)
        return property_function
    
    @predicate
    def infer(self, question: QuaRelType, answer_1: QuaRelType, answer_2: QuaRelType) -> Number:
        for theory in self.default_theories:
            if question.prop.name in theory and answer_1.prop.name in theory:
                world_same = 1 if question.world.number == answer_1.world.number else -1
                direction_same = 1 if question.direction.number == answer_1.direction.number else -1
                if theory[question.prop.name] * theory[answer_1.prop.name] * world_same * direction_same == 1:
                    return Number(1)
                else:
                    return Number(2)
            
            if question.prop.name in theory and answer_2.prop.name in theory:
                world_same = 1 if question.world.number == answer_2.world.number else -1
                direction_same = 1 if question.direction.number == answer_2.direction.number else -1
                if theory[question.prop.name] * theory[answer_2.prop.name] * world_same * direction_same == 1:
                    return Number(2)
                else:
                    return Number(1)
        return Number(1)

class QuaRelLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.language = QuaRel()
    
    def test_constant_logical_form(self):
        print('execute', self.language.execute('world1'))
            
    def test_infer_logical_form(self):
        self.language.execute('(infer (speed higher world1) (friction higher world1) (friction lower world1))')
        print(self.language.logical_form_to_action_sequence('(infer (speed higher world1) (friction higher world1) (friction lower world1))'))





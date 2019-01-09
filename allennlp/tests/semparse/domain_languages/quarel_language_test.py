from typing import Callable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import DomainLanguage, predicate

class Property():
    def __init__(self, name: str) -> None:
        self.name = name

class World():
    def __init__(self, number: int) -> None:
        self.number = number

class Direction():
    def __init__(self, number: int) -> None:
        self.number = number

class QuaRelType():
    def __init__(self, prop: Property, direction: Direction, world: World):
        self.prop = prop
        self.direction = direction
        self.world = world

def make_property_predicate(property_name: str) -> Callable[[Direction, World], QuaRelType]:
    def property_function(direction: Direction, world: World) -> QuaRelType:
        return QuaRelType(Property(property_name), direction, world)
    return property_function

class QuaRel(DomainLanguage):
    def __init__(self):
        super().__init__(start_types={int}, allowed_constants={'world1': World(1),
                                                               'world2': World(2),
                                                               'higher': Direction(1),
                                                               'lower': Direction(-1),
                                                               'high': Direction(1),
                                                               'low': Direction(-1)})
        self.default_theories = [{"friction": 1, "speed": -1, "smoothness": -1, "distance": -1, "heat": 1},
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

        for prop in ["friction", "speed", "distance", "heat", "smoothness", "acceleration",
                     "amountSweat", "apparentSize", "breakability", "brightness", "exerciseIntensity",
                     "flexibility", "gravity", "loudness", "mass", "strength", "thickness", "time", "weight"]:
            func = make_property_predicate(prop)
            self.add_predicate(prop, func)

        # ``and`` is a reserved word, so we add it as a predicate here instead of using the decorator.
        def and_function(quarel_0: QuaRelType, quarel_1: QuaRelType) -> QuaRelType:
            if self._check_compatible(quarel_0, quarel_1):
                return quarel_0
            else:
                return None
        self.add_predicate('and', and_function)

    def _check_compatible(self, quarel_0: QuaRelType, quarel_1: QuaRelType) -> bool:
        for theory in self.default_theories:
            if quarel_0.prop.name in theory and quarel_1.prop.name in theory:
                world_same = 1 if quarel_0.world.number == quarel_1.world.number else -1
                direction_same = 1 if quarel_0.direction.number == quarel_1.direction.number else -1
                if theory[quarel_0.prop.name] * theory[quarel_1.prop.name] * world_same * direction_same == 1:
                    return True
                else:
                    return False
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

    def test_infer_quarel(self):
        assert self.language.execute(('(infer (speed higher world1) (friction higher world1) '
                                      '(friction lower world1))')) == 1
        assert self.language.logical_form_to_action_sequence(('(infer (speed higher world1) '
                                                              '(friction higher world1) '
                                                              '(friction lower world1))')) == \
                ['@start@ -> int',
                 'int -> [<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, '
                 'QuaRelType]',
                 '<QuaRelType,QuaRelType,QuaRelType:int> -> infer',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> speed',
                 'Direction -> higher',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> higher',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> lower',
                 'World -> world1']

        assert self.language.execute(('(infer (speed higher world2) (friction higher world1) '
                                      '(friction lower world1))')) == 0

    def test_infer_quaval(self):
        assert self.language.execute(('(infer (and (thickness low world1) '
                                      '(thickness high world2)) '
                                      '(strength lower world1) '
                                      '(strength lower world2))')) == 0
        assert self.language.logical_form_to_action_sequence(('(infer (and (thickness low world1) '
                                                              '(thickness high world2)) '
                                                              '(strength lower world1) '
                                                              '(strength lower world2))')) == \
                ['@start@ -> int',
                 'int -> [<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, '
                 'QuaRelType]',
                 '<QuaRelType,QuaRelType,QuaRelType:int> -> infer',
                 'QuaRelType -> [<QuaRelType,QuaRelType:QuaRelType>, QuaRelType, QuaRelType]',
                 '<QuaRelType,QuaRelType:QuaRelType> -> and',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> thickness',
                 'Direction -> low',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> thickness',
                 'Direction -> high',
                 'World -> world2',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> strength',
                 'Direction -> lower',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> strength',
                 'Direction -> lower',
                 'World -> world2']

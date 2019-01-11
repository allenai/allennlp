"""
This module defines a domain language for the QuaRel dataset, a simple domain theory for reasoning
about qualitative relations.
"""
from typing import Callable

from allennlp.semparse.domain_languages.domain_language import DomainLanguage, predicate


class Property:
    def __init__(self, name: str) -> None:
        self.name = name


class World:
    def __init__(self, number: int) -> None:
        self.number = number


class Direction:
    def __init__(self, number: int) -> None:
        self.number = number


class QuaRelType:
    def __init__(self,
                 quarel_property: Property,
                 direction: Direction,
                 world: World) -> None:
        self.quarel_property = quarel_property
        self.direction = direction
        self.world = world


def make_property_predicate(property_name: str) -> Callable[[Direction, World], QuaRelType]:
    def property_function(direction: Direction, world: World) -> QuaRelType:
        return QuaRelType(Property(property_name), direction, world)
    return property_function


class QuaRelLanguage(DomainLanguage):
    """
    Domain language for the QuaRel dataset.
    """
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

        for quarel_property in ["friction", "speed", "distance", "heat", "smoothness", "acceleration",
                                "amountSweat", "apparentSize", "breakability", "brightness", "exerciseIntensity",
                                "flexibility", "gravity", "loudness", "mass", "strength", "thickness",
                                "time", "weight"]:
            func = make_property_predicate(quarel_property)
            self.add_predicate(quarel_property, func)

        # ``and`` is a reserved word, so we add it as a predicate here instead of using the decorator.
        def and_function(quarel_0: QuaRelType, quarel_1: QuaRelType) -> QuaRelType:
            # If the two relations are compatible, then we can return either of them.
            if self._check_quarels_compatible(quarel_0, quarel_1):
                return quarel_0
            else:
                return None
        self.add_predicate('and', and_function)

    def _check_quarels_compatible(self, quarel_0: QuaRelType, quarel_1: QuaRelType) -> bool:
        if not (quarel_0 and quarel_1):
            return False
        for theory in self.default_theories:
            if quarel_0.quarel_property.name in theory and quarel_1.quarel_property.name in theory:
                world_same = 1 if quarel_0.world.number == quarel_1.world.number else -1
                direction_same = 1 if quarel_0.direction.number == quarel_1.direction.number else -1
                is_compatible = theory[quarel_0.quarel_property.name] * theory[quarel_1.quarel_property.name] \
                        * world_same * direction_same
                if is_compatible == 1: # pylint: disable=simplifiable-if-statement
                    return True
                else:
                    return False
        return False

    @predicate
    def infer(self, setup: QuaRelType, answer_0: QuaRelType, answer_1: QuaRelType) -> int:
        """
        Take the question and check if it is compatible with either of the answer choices.
        """
        if self._check_quarels_compatible(setup, answer_0):
            if self._check_quarels_compatible(setup, answer_1):
                # Found two answers
                return -2
            else:
                return 0
        elif self._check_quarels_compatible(setup, answer_1):
            return 1
        else:
            return -1

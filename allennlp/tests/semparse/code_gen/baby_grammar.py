
from typing import Tuple, NamedTuple
from allennlp.common.registrable import Registrable

"""
A baby grammar to experiment with generating a parser specification
directly from python.

Constraints:
1. Everything must have type hints to work. Union types are not allowed.
2. Classes are permitted, but inheritance between classes
is strict, in the sense that super classes must implement
no more functionality than the base class. TODO(Mark): is this required?
"""

class Box:
    def __init__(self, height: int, width: int) -> None:
        self._height: int = height
        self._width: int = width

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

def height(x: Box) -> int:
    return x._height

def expand(x: Box, value: int) -> Box:
    return Box(x.height() + value, x.width() + value)

def area(b: Box) -> int:
    return b.height() * b.width()

def perimeter(b: Box) -> int:
    return (2 * b.height()) + (2 * b.width())

def add(x: int, y: int) -> int:
    return x + y

def subtract(x: int, y: int) -> int:
    return x - y

top_level_grammar = {expand, area, perimeter, subtract}

examples = {
    "Please can you expand this box of size (2,3) by 5?": expand(Box(2,3), 5),
    "What is the area of a box with height 2 and width 3?": area(Box(height=2, width=3)),
    "What is the difference between the areas of a box (2,3) and a box (4,5)": subtract(area(Box(height=2, width=3)), area(Box(height=4, width=5)))
}


__all__ = ["Box", "expand", "area", "perimeter", "add", "subtract"]
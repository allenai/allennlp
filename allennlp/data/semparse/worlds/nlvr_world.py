"""
This module defines three classes: Object and Box (the two entities in the NLVR domain) and a NLVRWorld,
which mainly contains an execution method and related helper methods.
"""

from collections import defaultdict
import operator

from typing import List, Dict, Set, Callable, Union, TypeVar

import pyparsing


AttributeType = Union[int, str]  # pylint: disable=invalid-name


class Object:
    """
    ``Objects`` are the geometric shapes in the NLVR domain. They have values for attributes shape, color,
    x_loc, y_loc and size. We take a dict read from the json file and store it here, and define a get method
    for getting the attribute values. We need this to be hashable because need to make sets of ``Objects``
    during execution, which get passed around between functions.

    Parameters
    ----------
    object_dict : Dict[str: Union[str, int]]
        The dict for each object from the json file.
    """
    def __init__(self, object_dict: Dict[str: AttributeType]) -> None:
        self._object_dict: Dict[str, AttributeType] = {}
        for key, value in object_dict.items():
            if isinstance(value, str):
                # The dataset has a hex code only for blue for some reason.
                if key == "color" and value.startswith("#"):
                    self._object_dict[key] = "blue"
                else:
                    self._object_dict[key] = value.lower()
            else:
                self._object_dict[key] = value

    def get_attribute(self, attribute):
        return self._object_dict[attribute]

    def __str__(self):
        color = self.get_attribute("color")
        shape = self.get_attribute("type")
        x_loc = self.get_attribute("x_loc")
        y_loc = self.get_attribute("y_loc")
        return "%s %s at (%d, %d)" % (color, shape, x_loc, y_loc)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Box:
    """
    This class represents each box containing objects in NLVR.

    Parameters
    ----------
    name : str
        This is just so that we can define a string representation to hash each object. It could be any unique
        string.
    objects_list : List[Dict]
        List of objects in the box, as given by the json file.
    """
    def __init__(self, name: str, objects_list: List[Dict]) -> None:
        self._name = name
        self._objects_set = set([Object(object_dict) for object_dict in objects_list])

    def get_objects(self) -> Set[Object]:
        return self._objects_set

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


EntityType = TypeVar('EntityType', Object, Box)  # pylint: disable=invalid-name


class NLVRWorld:
    # pylint: disable=too-many-public-methods
    """
    Class defining the world representation of NLVR. Defines an execution logic for logical forms in NLVR.
    We just take the structured_rep from the json file to initialize this.

    Parameters
    ----------
    world_representation : List[List[Dict]]
        "structured_rep" from the json file.
    """
    def __init__(self, world_representation: List[List[Dict]]) -> None:
        self._boxes = set([Box("box%d" % index, object_list) for index, object_list in
                           enumerate(world_representation)])
        self._objects: Set[Object] = set()
        for box in self._boxes:
            self._objects = self._objects.union(box.get_objects())

    def all_boxes(self) -> Set[Box]:
        return self._boxes

    def all_objects(self) -> Set[Object]:
        return self._objects

    def execute(self, logical_form: str) -> bool:
        """
        Execute the logical form.
        """
        def _flatten(nested_list, acc):
            if isinstance(nested_list, list):
                for item in nested_list:
                    acc = _flatten(item, acc)
                return acc
            acc.append(nested_list)
            return acc

        def _apply_function_list(function_list, arg):
            return_value = getattr(self, function_list[-1])(arg)
            for function_name in reversed(function_list[:-1]):
                return_value = getattr(self, function_name)(return_value)
            return return_value

        def _execute_sub_expression(sub_expression):
            # pylint: disable=too-many-return-statements
            if isinstance(sub_expression, list) and len(sub_expression) == 1:
                # List with a single item. We should just evaluate the item.
                return _execute_sub_expression(sub_expression[0])

            if isinstance(sub_expression, list):
                # We have to specially deal with box filtering functions and assertion functions, each of which
                # takes multiple arguments. Let's do that first
                function = getattr(self, sub_expression[0])
                if sub_expression[0].startswith('filter_'):
                    # filter_* functions are box filtering functions, and have a lambda expression as the
                    # second argument. We'll process the whole nested structure of the lambda expression
                    # here.
                    arguments = sub_expression[1]
                    set_to_filter = _execute_sub_expression(arguments[0])
                    flattened_lambda_terms = _flatten(arguments[1:-1], [])
                    attribute_function = lambda x: _apply_function_list(flattened_lambda_terms, x)
                    attribute = arguments[-1]
                    return function(set_to_filter, attribute_function, attribute)
                elif sub_expression[0].startswith('assert_'):
                    # assert functions are the highest level boolean functions. They take two arguments,
                    # that evaluate to strings or integers, and compare them.
                    arguments = sub_expression[1]
                    first_attribute = _execute_sub_expression(arguments[:2])
                    second_attribute = _execute_sub_expression(arguments[2])
                    return function(first_attribute, second_attribute)
                elif isinstance(sub_expression[0], str) and isinstance(sub_expression[1], list):
                    # These are the other kinds of function applications.
                    arguments = _execute_sub_expression(sub_expression[1])
                    if isinstance(arguments, set):
                        # This means arguments is an execution of ``all_objects`` or ``all_boxes``
                        return function(arguments)
                    # Or else, arguments are an actual list of executed arguments.
                    return function(*arguments)
                raise RuntimeError("Invalid subtree: %s" % sub_expression)

            if isinstance(sub_expression, str):
                # These can either be numbers, shapes, colors or the special sets (all_objects or all_boxes)
                if str.isdigit(sub_expression):
                    return int(sub_expression)
                elif sub_expression.startswith('color_'):
                    return sub_expression.replace('color_', '')
                elif sub_expression.startswith('shape_'):
                    return sub_expression.replace('shape_', '')
                # This has to be all_objects or all_boxes
                return getattr(self, sub_expression)()

        if not logical_form.startswith("("):
            logical_form = "(%s)" % logical_form
        logical_form = logical_form.replace(",", " ")
        expression_as_list = pyparsing.OneOrMore(pyparsing.nestedExpr()).parseString(logical_form).asList()

        return _execute_sub_expression(expression_as_list)

    @staticmethod
    def color(objects_set: Set[Object]) -> Set[str]:
        """
        Returns the set of colors of a set of objects.
        """
        return set([obj.get_attribute("color") for obj in objects_set])

    @classmethod
    def _get_single_color(cls, _object: Object) -> str:
        # ``color`` takes a set of objects and returns a set of colors. We often want to get the color of a
        # single object. This method does it.
        return list(cls.color(set([_object])))[0]

    @staticmethod
    def shape(objects_set: Set[Object]) -> Set[str]:
        """
        Returns the set of shapes of a set of objects.
        """
        return set([obj.get_attribute("type") for obj in objects_set])

    @classmethod
    def _get_single_shape(cls, _object: Object) -> str:
        # ``shape`` takes a set of objects and returns a set of shapes. We often want to get the shape of a
        # single object. This method does it.
        return list(cls.shape(set([_object])))[0]

    @staticmethod
    def count(entities_set: Set[EntityType]) -> int:
        return len(entities_set)

    @staticmethod
    def object_in_box(box: Box) -> Set[Object]:
        return box.get_objects()

    @staticmethod
    def _filter(set_to_filter: Set[EntityType],
                attribute_function: Callable[[EntityType], AttributeType],
                target_attribute: AttributeType,
                comparison_op: Callable[[AttributeType, AttributeType], bool]) -> Set[EntityType]:
        returned_set = []
        for entity in set_to_filter:
            if comparison_op(attribute_function(entity), target_attribute):
                returned_set.append(entity)
        return set(returned_set)

    @classmethod
    def filter_equals(cls,
                      set_to_filter: Set[Box],
                      attribute_function: Callable[[Box], AttributeType],
                      target_attribute: AttributeType) -> Set[Box]:
        return cls._filter(set_to_filter, attribute_function, target_attribute, operator.eq)

    @classmethod
    def filter_not_equal(cls,
                         set_to_filter: Set[Box],
                         attribute_function: Callable[[Box], AttributeType],
                         target_attribute: AttributeType) -> Set[Box]:
        return cls._filter(set_to_filter, attribute_function, target_attribute, operator.ne)

    @classmethod
    def filter_greater_equal(cls,
                             set_to_filter: Set[Box],
                             attribute_function: Callable[[Box], AttributeType],
                             target_attribute: AttributeType) -> Set[Box]:
        return cls._filter(set_to_filter, attribute_function, target_attribute, operator.ge)

    @classmethod
    def filter_lesser_equal(cls,
                            set_to_filter: Set[Box],
                            attribute_function: Callable[[Box], AttributeType],
                            target_attribute: AttributeType) -> Set[Box]:
        return cls._filter(set_to_filter, attribute_function, target_attribute, operator.le)

    @classmethod
    def black(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, cls._get_single_color, 'black', operator.eq)

    @classmethod
    def blue(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, cls._get_single_color, 'blue', operator.eq)

    @classmethod
    def yellow(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, cls._get_single_color, 'yellow', operator.eq)

    @classmethod
    def circle(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, cls._get_single_shape, 'circle', operator.eq)

    @classmethod
    def square(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, cls._get_single_shape, 'square', operator.eq)

    @classmethod
    def triangle(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, cls._get_single_shape, 'triangle', operator.eq)

    @classmethod
    def same_color(cls, objects_set: Set[Object]) -> Set[Object]:
        """
        Returns the set of objects whose color is the most frequent color in the initial set, if the highest
        frequency is greater than 1. If not, all objects are of different colors, and this method returns a
        null set.
        """
        all_colors = [cls._get_single_color(obj) for obj in objects_set]
        color_counts: Dict[str, int] = defaultdict(int)
        for _color in all_colors:
            color_counts[_color] += 1
        if max(color_counts.values()) <= 1:
            return set()
        most_frequent_color = max(all_colors, key=color_counts.get)
        return cls._filter(objects_set, cls._get_single_color, most_frequent_color, operator.eq)

    @classmethod
    def same_shape(cls, objects_set: Set[Object]) -> Set[Object]:
        """
        Returns the set of objects whose shape is the most frequent shape in the initial set, if the highest
        frequency is greater than 1. If not, all objects are of different shapes, and this method returns a
        null set.
        """
        all_shapes = [cls._get_single_shape(obj) for obj in objects_set]
        shape_counts: Dict[str, int] = defaultdict(int)
        for _shape in all_shapes:
            shape_counts[_shape] += 1
        if max(shape_counts.values()) <= 1:
            return set()
        most_frequent_shape = max(all_shapes, key=shape_counts.get)
        return cls._filter(objects_set, cls._get_single_shape, most_frequent_shape, operator.eq)

    @classmethod
    def touch_bottom(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("y_loc"), 0, operator.eq)

    @classmethod
    def touch_left(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("x_loc"), 0, operator.eq)

    @classmethod
    def touch_top(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("y_loc") + x.get_attribute("size"),
                           100, operator.eq)

    @classmethod
    def touch_right(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("x_loc") + x.get_attribute("size"),
                           100, operator.eq)

    @classmethod
    def touch_wall(cls, objects_set: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(cls.touch_top(objects_set), cls.touch_left(objects_set),
                                cls.touch_right(objects_set), cls.touch_bottom(objects_set))

    @classmethod
    def touch_corner(cls, objects_set: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(cls.touch_top(objects_set).intersection(cls.touch_right(objects_set)),
                                cls.touch_top(objects_set).intersection(cls.touch_left(objects_set)),
                                cls.touch_bottom(objects_set).intersection(cls.touch_right(objects_set)),
                                cls.touch_bottom(objects_set).intersection(cls.touch_left(objects_set)))

    @classmethod
    def small(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("size"), 10, operator.eq)

    @classmethod
    def medium(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("size"), 20, operator.eq)

    @classmethod
    def big(cls, objects_set: Set[Object]) -> Set[Object]:
        return cls._filter(objects_set, lambda x: x.get_attribute("size"), 30, operator.eq)

    @staticmethod
    def negate_filter(filter_function: Callable[[Set[EntityType]], Set[EntityType]],
                      objects_set: Set[EntityType]) -> Set[EntityType]:
        return objects_set.difference(filter_function(objects_set))

    @staticmethod
    def assert_equals(actual_attribute: AttributeType, target_attribute: AttributeType) -> bool:
        return actual_attribute == target_attribute

    @staticmethod
    def assert_not_equal(actual_attribute: AttributeType, target_attribute: AttributeType) -> bool:
        return actual_attribute != target_attribute

    @staticmethod
    def assert_greater(actual_attribute: int, target_attribute: int) -> bool:
        return actual_attribute > target_attribute

    @staticmethod
    def assert_lesser(actual_attribute: int, target_attribute: int) -> bool:
        return actual_attribute < target_attribute

    @staticmethod
    def assert_greater_equals(actual_attribute: int, target_attribute: int) -> bool:
        return actual_attribute >= target_attribute

    @staticmethod
    def assert_lesser_equals(actual_attribute: int, target_attribute: int) -> bool:
        return actual_attribute <= target_attribute

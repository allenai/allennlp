"""
This module defines classes Object and Box (the two entities in the NLVR domain) and a NLVRWorld,
which mainly contains an execution method and related helper methods.
"""

from collections import defaultdict
import operator
from typing import Any, List, Dict, Set, Callable, Union
import pyparsing

from allennlp.common.util import JsonDict


AttributeType = Union[int, str]  # pylint: disable=invalid-name


class Object:
    """
    ``Objects`` are the geometric shapes in the NLVR domain. They have values for attributes shape, color,
    x_loc, y_loc and size. We take a dict read from the json file and store it here, and define a get method
    for getting the attribute values. We need this to be hashable because need to make sets of ``Objects``
    during execution, which get passed around between functions.

    Parameters
    ----------
    object_dict : Dict[str, Union[str, int]]
        The dict for each object from the json file.
    """
    def __init__(self, object_dict: JsonDict) -> None:
        for key, value in object_dict.items():
            if key == "color":
                # The dataset has a hex code only for blue for some reason.
                if value.startswith("#"):
                    self.color = "blue"
                else:
                    self.color = value.lower()
            elif key == "type":
                self.shape = value.lower()
            elif key == "x_loc":
                self.x_loc = int(value)
            elif key == "y_loc":
                self.y_loc = int(value)
            elif key == "size":
                self.size = int(value)

    def __str__(self):
        return "%s %s at (%d, %d)" % (self.color, self.shape, self.x_loc, self.y_loc)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Box:
    """
    This class represents each box containing objects in NLVR.

    Parameters
    ----------
    objects_list : List[JsonDict]
        List of objects in the box, as given by the json file.
    name : str (optional)
        Optionally specify a string representation. It could be any unique string. If not specified, we will use
        the list of object names.
    """
    def __init__(self,
                 objects_list: List[JsonDict],
                 name: str = None) -> None:
        self.objects = set([Object(object_dict) for object_dict in objects_list])
        self._name = name or str([str(obj) for obj in objects_list])

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class NLVRWorld:
    # pylint: disable=too-many-public-methods
    """
    Class defining the world representation of NLVR. Defines an execution logic for logical forms in NLVR.
    We just take the structured_rep from the json file to initialize this.

    Parameters
    ----------
    world_representation : JsonDict
        structured_rep from the json file.
    """
    def __init__(self, world_representation: List[List[JsonDict]]) -> None:
        self._boxes = set([Box(object_list, "box%d" % index)
                           for index, object_list in enumerate(world_representation)])
        self._objects: Set[Object] = set()
        for box in self._boxes:
            self._objects.update(box.objects)

    def all_boxes(self) -> Set[Box]:
        return self._boxes

    def all_objects(self) -> Set[Object]:
        return self._objects

    @classmethod
    def _flatten(cls,
                 nested_list: Union[str, Union[str, List[str]]],
                 flat_list: List[str]) -> List[str]:
        """
        Take a nested list and make it a flat list.
        """
        if isinstance(nested_list, list):
            for item in nested_list:
                flat_list = cls._flatten(item, flat_list)
            return flat_list
        flat_list.append(nested_list)
        return flat_list

    def _apply_function_list(self,
                             function_list: List[str],
                             argument: Any) -> Any:
        """
        Take a flat list of functions and an argument and apply them iteratively in reverse order.
        """
        return_value = argument
        for function_name in reversed(function_list):
            return_value = getattr(self, function_name)(return_value)
        return return_value

    def execute(self, logical_form: str) -> bool:
        """
        Execute the logical form.
        The language we defined here contains six types of functions, four of which return sets, one returns
        integers and one returns booleans.

        1) Attribute functions - These are of the form `attribute(set_of_boxes_or_objects)`. They take sets and
        return sets of attributes. `color` and `shape` (operating on objects) and `object_in_box` (operating on
        boxes) are the attribute functions.

        2) Count function - Takes a set of objects or boxes and returns its length.

        3) Box filtering functions - These are of the form
        `filter(set_of_boxes, attribute_function, target_attribute)`
        The idea is that we take a set of boxes, an attribute function that extracts the relevant attribute from
        a box, and a target attribute that we compare against. The logic is that we execute the attribute function
        on _each_ of the given boxes and return only those whose attribute value, in comparison with the target
        attribute, satisfies the filtering criterion (i.e., equal to the target, less than, greater than etc.). The
        fitering function defines the comparison operator.
        All the functions below with names `filter_*` belong to this category.

        4) Object filtering functions - These are of the form `filter(set_of_objects)`. These are similar to box
        filtering functions, but they operate on objects instead. Also, note that they take just one argument
        instead of three. This is because while box filtering functions typically query complex attributes, object
        filtering functions query the properties of the objects alone. These are simple and finite in number. Thus,
        we essentially let the filtering function define the attribute function, and the target attribute as well,
        along with the comparison operator. That is, these are functions like `black` (which takes a set of
        objects, and returns those whose "color" (attribute function) "equals" (comparison operator) "black"
        (target attribute)), or "square" (which returns objects that are squares).

        5) Negate object filter - Takes an object filter and a set of objects and applies the negation of the
        object filter on the set.

        6) Assert operations - These typically occur only at the root node of the logical form trees. They take a
        value (obtained from a filtering operation), compare it against a target and return True or False. All the
        functions that have names like `assert_*` are assert functions.
        """
        def _execute_sub_expression(sub_expression: Union[str, List[str]]) -> Union[int, str, bool, Set[Object]]:
            """
            Primary recursive method for execution.
            """
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
                    flattened_lambda_terms = self._flatten(arguments[1:-1], [])
                    attribute_function = lambda x: self._apply_function_list(flattened_lambda_terms, x)
                    attribute = _execute_sub_expression(arguments[-1])
                    return function(set_to_filter, attribute_function, attribute)
                elif sub_expression[0].startswith('assert_'):
                    # assert functions are the highest level boolean functions. They take two arguments,
                    # that evaluate to strings or integers, and compare them.
                    arguments = sub_expression[1]
                    first_attribute = _execute_sub_expression(arguments[:2])
                    second_attribute = _execute_sub_expression(arguments[2])
                    return function(first_attribute, second_attribute)
                elif sub_expression[0] == "negate_filter":
                    arguments = sub_expression[1]
                    original_filter = getattr(self, arguments[0])
                    initial_set: Set[Object] = _execute_sub_expression(arguments[1:])
                    return self.negate_filter(original_filter, initial_set)
                elif isinstance(sub_expression[0], str) and isinstance(sub_expression[1], list):
                    # These are the other kinds of function applications.
                    arguments = _execute_sub_expression(sub_expression[1])
                    if isinstance(arguments, set):
                        # This means arguments is an execution of all_objects or all_boxes
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

    ## Attribute functions
    @staticmethod
    def color(objects: Set[Object]) -> Set[str]:
        """
        Returns the set of colors of a set of objects.
        """
        return set([obj.color for obj in objects])

    @staticmethod
    def shape(objects: Set[Object]) -> Set[str]:
        """
        Returns the set of shapes of a set of objects.
        """
        return set([obj.shape for obj in objects])

    @staticmethod
    def count(entities_set: Set[Union[Object, Box]]) -> int:
        return len(entities_set)

    @staticmethod
    def object_in_box(box: Box) -> Set[Object]:
        return box.objects

    @staticmethod
    def _filter_boxes(set_to_filter: Set[Box],
                      attribute_function: Callable[[Box], AttributeType],
                      target_attribute: AttributeType,
                      comparison_op: Callable[[AttributeType, AttributeType], bool]) -> Set[Box]:
        returned_set = set()
        for entity in set_to_filter:
            if comparison_op(attribute_function(entity), target_attribute):
                returned_set.add(entity)
        return returned_set

    ## Box filtering functions
    @classmethod
    def filter_equals(cls,
                      set_to_filter: Set[Box],
                      attribute_function: Callable[[Box], AttributeType],
                      target_attribute: AttributeType) -> Set[Box]:
        return cls._filter_boxes(set_to_filter, attribute_function, target_attribute, operator.eq)

    @classmethod
    def filter_not_equal(cls,
                         set_to_filter: Set[Box],
                         attribute_function: Callable[[Box], AttributeType],
                         target_attribute: AttributeType) -> Set[Box]:
        return cls._filter_boxes(set_to_filter, attribute_function, target_attribute, operator.ne)

    @classmethod
    def filter_greater_equal(cls,
                             set_to_filter: Set[Box],
                             attribute_function: Callable[[Box], AttributeType],
                             target_attribute: AttributeType) -> Set[Box]:
        return cls._filter_boxes(set_to_filter, attribute_function, target_attribute, operator.ge)

    @classmethod
    def filter_lesser_equal(cls,
                            set_to_filter: Set[Box],
                            attribute_function: Callable[[Box], AttributeType],
                            target_attribute: AttributeType) -> Set[Box]:
        return cls._filter_boxes(set_to_filter, attribute_function, target_attribute, operator.le)

    ## Object filtering functions
    @classmethod
    def black(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.color == "black"])

    @classmethod
    def blue(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.color == "blue"])

    @classmethod
    def yellow(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.color == "yellow"])

    @classmethod
    def circle(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.shape == "circle"])

    @classmethod
    def square(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.shape == "square"])

    @classmethod
    def triangle(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.shape == "triangle"])

    @staticmethod
    def _get_objects_with_same_attribute(objects: Set[Object],
                                         attribute_function: Callable[[Object], str]) -> Set[Object]:
        """
        Returns the set of objects for which the attribute function returns an attribute value that is most
        frequent in the initial set, if the frequency is greater than 1. If not, all objects have different
        attribute values, and this method returns an empty set.
        """
        objects_of_attribute: Dict[str, Set[Object]] = defaultdict(set)
        for entity in objects:
            objects_of_attribute[attribute_function(entity)].add(entity)
        most_frequent_attribute = max(objects_of_attribute, key=lambda x: len(objects_of_attribute[x]))
        if len(objects_of_attribute[most_frequent_attribute]) <= 1:
            return set()
        return objects_of_attribute[most_frequent_attribute]

    @classmethod
    def same_color(cls, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent color in the initial
        set of objects, if the highest frequency is greater than 1, or an empty set otherwise.

        This is an unusual name for what the method does, but just as ``blue`` filters objects to those that are
        blue, this filters objects to those that are of the same color.
        """
        return cls._get_objects_with_same_attribute(objects, lambda x: x.color)

    @classmethod
    def same_shape(cls, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent color in the initial
        set of objects, if the highest frequency is greater than 1, or an empty set otherwise.

        This is an unusual name for what the method does, but just as ``triangle`` filters objects to
        those that are triangles, this filters objects to those that are of the same shape.
        """
        return cls._get_objects_with_same_attribute(objects, lambda x: x.shape)

    @classmethod
    def touch_bottom(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.y_loc == 0])

    @classmethod
    def touch_left(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.x_loc == 0])

    @classmethod
    def touch_top(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.y_loc + obj.size == 100])

    @classmethod
    def touch_right(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.x_loc + obj.size == 100])

    @classmethod
    def touch_wall(cls, objects: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(cls.touch_top(objects), cls.touch_left(objects),
                                cls.touch_right(objects), cls.touch_bottom(objects))

    @classmethod
    def touch_corner(cls, objects: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(cls.touch_top(objects).intersection(cls.touch_right(objects)),
                                cls.touch_top(objects).intersection(cls.touch_left(objects)),
                                cls.touch_bottom(objects).intersection(cls.touch_right(objects)),
                                cls.touch_bottom(objects).intersection(cls.touch_left(objects)))

    @classmethod
    def small(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.size == 10])

    @classmethod
    def medium(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.size == 20])

    @classmethod
    def big(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.size == 30])

    @staticmethod
    def negate_filter(filter_function: Callable[[Set[Object]], Set[Object]],
                      objects: Set[Object]) -> Set[Object]:
        # Negate an object filter.
        return objects.difference(filter_function(objects))

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

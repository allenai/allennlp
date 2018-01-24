"""
This module defines classes Object and Box (the two entities in the NLVR domain) and an NlvrWorld,
which mainly contains an execution method and related helper methods.
"""
from collections import defaultdict
import operator
from typing import Any, List, Dict, Set, Callable, TypeVar, Union

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.common import util
from allennlp.common.util import JsonDict
from allennlp.data.semparse import util as semparse_util
from allennlp.data.semparse.type_declarations import nlvr_type_declaration as types
from allennlp.data.semparse.worlds.world import World


AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


class Object:
    """
    ``Objects`` are the geometric shapes in the NLVR domain. They have values for attributes shape,
    color, x_loc, y_loc and size. We take a dict read from the JSON file and store it here, and
    define a get method for getting the attribute values. We need this to be hashable because need
    to make sets of ``Objects`` during execution, which get passed around between functions.

    Parameters
    ----------
    attributes : ``JsonDict``
        The dict for each object from the json file.
    """
    def __init__(self, attributes: JsonDict) -> None:
        object_color = attributes["color"]
        # The dataset has a hex code only for blue for some reason.
        if object_color.startswith("#"):
            self.color = "blue"
        else:
            self.color = object_color.lower()
        self.shape = attributes["type"].lower()
        self.x_loc = attributes["x_loc"]
        self.y_loc = attributes["y_loc"]
        self.size = attributes["size"]

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
    objects_list : ``List[JsonDict]``
        List of objects in the box, as given by the json file.
    name : ``str`` (optional)
        Optionally specify a string representation. It could be any unique string. If not
        specified, we will use the list of object names.
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


class NlvrWorld(World):
    """
    Class defining the world representation of NLVR. Defines an execution logic for logical forms
    in NLVR.  We just take the structured_rep from the JSON file to initialize this.

    Parameters
    ----------
    world_representation : ``JsonDict``
        structured_rep from the JSON file.
    """
    # pylint: disable=too-many-public-methods
    # TODO(pradeep): Define more spatial relationship methods: above, below, left_of, right_of..
    def __init__(self, world_representation: List[List[JsonDict]]) -> None:
        super(NlvrWorld, self).__init__(global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                        global_name_mapping=types.COMMON_NAME_MAPPING)
        self._boxes = set([Box(object_list, "box%d" % index)
                           for index, object_list in enumerate(world_representation)])
        self._objects: Set[Object] = set()
        for box in self._boxes:
            self._objects.update(box.objects)

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        return types.COMMON_NAME_MAPPING[name] if name in types.COMMON_NAME_MAPPING else name

    def _apply_function_list(self,
                             functions: List[str],
                             argument: Any) -> Any:
        """
        Take a flat list of functions in ``NlvrWorld`` and an argument and apply them iteratively in reverse order.
        """
        return_value = argument
        for function_name in reversed(functions):
            return_value = getattr(self, function_name)(return_value)
        return return_value

    def execute(self, logical_form: str) -> bool:
        """
        Execute the logical form. The top level function is an assertion function (see below). We
        just parse the string into a list and pass the whole thing to ``_execute_assertion`` and
        let the method deal with it. This is because the dataset contains sentences (instead of questions), and
        they evaluate to either true or false.

        The language we defined here contains seven types of functions, five of which return sets,
        and one returns integers, and one returns booleans.

        1) Assertion Function : These occur only at the root node of the logical form trees. They
        take a value obtained from an attribute function (see "Attribute Functions" below), compare
        it against a target and return True or False. All the functions that have names like
        `assert_*` are assert functions.

        2) Attribute Functions : These are of the form ``attribute(set_of_boxes_or_objects)``. There
        are two types of attribute functions:

            2a) Object Attribute Functions: They take sets of objects and return sets of
            attributes.  `color`, `shape` are the attribute functions.

            2b) Count Function : Takes a set of objects or boxes and returns its cardinality.

        3) Box Membership Function : This takes a box as an argument and returns the objects in it.
        This is a special kind of attribute function for two reasons. Firstly, it returns a set of
        objects instead of attributes, and secondly it occurs only within the second argument of a
        box filtering function (see below). It provides a way to query boxes based on the
        attributes of objects contained within it. The function is called ``object_in_box``, and it
        gets executed within ``_execute_box_filter``.

        4) Box Filtering Functions : These are of the form `filter(set_of_boxes,
        attribute_function, target_attribute)` The idea is that we take a set of boxes, an
        attribute function that extracts the relevant attribute from a box, and a target attribute
        that we compare against. The logic is that we execute the attribute function on `each` of
        the given boxes and return only those whose attribute value, in comparison with the target
        attribute, satisfies the filtering criterion (i.e., equal to the target, less than, greater
        than etc.). The fitering function defines the comparison operator.  All the functions in
        this class with names ``filter_*`` belong to this category.

        5) Object Filtering Functions : These are of the form ``filter(set_of_objects)``. These are
        similar to box filtering functions, but they operate on objects instead. Also, note that
        they take just one argument instead of three. This is because while box filtering functions
        typically query complex attributes, object filtering functions query the properties of the
        objects alone.  These are simple and finite in number. Thus, we essentially let the
        filtering function define the attribute function, and the target attribute as well, along
        with the comparison operator.  That is, these are functions like `black` (which takes a set
        of objects, and returns those whose "color" (attribute function) "equals" (comparison
        operator) "black" (target attribute)), or "square" (which returns objects that are
        squares).

        6) Negate Object Filter : Takes an object filter and a set of objects and applies the
        negation of the object filter on the set.
        """
        if not logical_form.startswith("("):
            logical_form = "(%s)" % logical_form
        logical_form = logical_form.replace(",", " ")
        expression_as_list = semparse_util.lisp_to_nested_expression(logical_form)
        # The whole expression has to be an assertion expression because it has to return a boolean.
        # TODO(pradeep): May want to make this more general and let the executor deal with questions.
        return self._execute_assertion(expression_as_list)

    def _execute_assertion(self, sub_expression: List) -> bool:
        """
        Assertion functions are boolean functions. They take two arguments, which
        evaluate to strings or integers, and compare them. The first element in the input list
        should be the assertion function name, the second as an attribute function, and the
        last element should be a constant.

        Syntax: assertion_function(attribute_function, constant)
        """
        # TODO(pradeep): We may want to change the order of arguments here to make decoding easier.
        assert isinstance(sub_expression, list), "Invalid assertion expression: %s" % sub_expression
        if len(sub_expression) == 1 and isinstance(sub_expression[0], list):
            return self._execute_assertion(sub_expression[0])
        assert isinstance(sub_expression[0], str) and sub_expression[0].startswith("assert_"),\
               "Invalid assertion function: %s" % (sub_expression[0])
        function = getattr(self, sub_expression[0])
        attribute_function = sub_expression[1]
        target_attribute = self._execute_constant(sub_expression[2])
        if attribute_function[0] == "count":
            counted_value = self._execute_count_function(attribute_function)
            return function(counted_value, target_attribute)
        attribute_set = self._execute_attribute_function(attribute_function)
        # We defined attribute functions to always return sets. But when this assertion is being executed, for the
        # assertion to be true, the returned set needs to be singleton. If not, the assertion should fail.
        if len(attribute_set) != 1:
            return False
        (attribute,) = attribute_set
        return function(attribute, target_attribute)

    def _execute_attribute_function(self, sub_expression: List) -> Set[str]:
        """
        Attribute functions return a set of evaluated attributes. The function has to be ``color``
        or ``shape``, and the only element in the nested list should evaluate to a set of objects.

        Syntax: attribute_function(input_set)
        """
        assert isinstance(sub_expression, list), "Invalid attribute expression: %s" % sub_expression
        function_name = sub_expression[0]
        arguments_list = sub_expression[1]
        input_set = self._execute_object_filter(arguments_list)
        if function_name == "shape":
            return self.shape(input_set)
        elif function_name == "color":
            return self.color(input_set)
        else:
            raise RuntimeError("Invalid attribute function: %s" % sub_expression[0])

    def _execute_count_function(self, sub_expression: List) -> int:
        """
        Acceptable syntax is count(object_set).
        """
        arguments_list = sub_expression[1]
        if arguments_list[0].startswith("filter_") or arguments_list[0] == "all_boxes":
            return self.count(self._execute_box_filter(arguments_list))
        else:
            return self.count(self._execute_object_filter(arguments_list))

    def _execute_box_filter(self, sub_expression: Union[str, List]) -> Set[Box]:
        """
        Box filtering functions either apply a filter on a set of boxes and return the filtered set,
        or return all the boxes.
        The elements should evaluate to one of the following:
            box_filtering_function(set_to_filter, attribute_function, constant)
            all_boxes
        """
        # TODO(pradeep): We may want to change the order of arguments here to make decoding easier.
        if sub_expression[0].startswith('filter_'):
            # filter_* functions are box filtering functions, and have a lambda expression as the
            # second argument. We'll process the whole nested structure of the lambda expression
            # here.
            function = getattr(self, sub_expression[0])
            set_to_filter = self._execute_box_filter(sub_expression[1])
            attribute_function_list = sub_expression[2]
            assert attribute_function_list[:2] == ['lambda', 'x'], ("Invalid lambda expression: %s" %
                                                                    attribute_function_list)
            # The list looks like ['lambda' 'x' ['f' ['g' ['x']]]]. We're flattening the nested list.
            flattened_lambda_terms = util.flatten_list(attribute_function_list[2])
            assert flattened_lambda_terms[-2:] == ['var', 'x'], ("Invalid lambda expression: %s" %
                                                                 attribute_function_list)
            attribute_function = lambda x: self._apply_function_list(flattened_lambda_terms[:-2], x)
            attribute = self._execute_constant(sub_expression[-1])
            return function(set_to_filter, attribute_function, attribute)
        elif sub_expression == 'all_boxes' or sub_expression[0] == 'all_boxes':
            return self._boxes
        else:
            raise RuntimeError("Invalid box filter expression: %s" % sub_expression)

    def _execute_object_filter(self, sub_expression: Union[str, List]) -> Set[Object]:
        """
        Object filtering functions should either be a string referring to all objects, or list which
        executes to a filtering operation.
        The elements should evaluate to one of the following:
            object_filtering_function(object_set)
            negate_filter(object_filtering_function, object_set)
            all_objects
        """
        if sub_expression[0] == "negate_filter":
            original_filter_name = sub_expression[1]
            if isinstance(original_filter_name, list):
                original_filter_name = original_filter_name[0]
            original_filter = getattr(self, original_filter_name)
            initial_set = self._execute_object_filter(sub_expression[2])
            return self.negate_filter(original_filter, initial_set)
        elif sub_expression == "all_objects" or sub_expression[0] == "all_objects":
            return self._objects
        elif isinstance(sub_expression[0], str) and isinstance(sub_expression[1], list):
            # These are functions like black, square, same_color etc.
            function = getattr(self, sub_expression[0])
            arguments = self._execute_object_filter(sub_expression[1])
            return function(arguments)
        else:
            raise RuntimeError("Invalid object filter expression: %s" % sub_expression)

    @staticmethod
    def _execute_constant(sub_expression: str) -> Union[str, int]:
        """
        Acceptable constants are numbers or strings starting with `shape_` or `color_`
        """
        if str.isdigit(sub_expression):
            return int(sub_expression)
        elif sub_expression.startswith('color_'):
            return sub_expression.replace('color_', '')
        elif sub_expression.startswith('shape_'):
            return sub_expression.replace('shape_', '')
        else:
            raise RuntimeError("Invalid constant: %s" % sub_expression)

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
    def count(entities_set: Union[Set[Box], Set[Object]]) -> int:
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
                             attribute_function: Callable[[Box], int],
                             target_attribute: int) -> Set[Box]:
        return cls._filter_boxes(set_to_filter, attribute_function, target_attribute, operator.ge)

    @classmethod
    def filter_lesser_equal(cls,
                            set_to_filter: Set[Box],
                            attribute_function: Callable[[Box], int],
                            target_attribute: int) -> Set[Box]:
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
        Returns the set of objects for which the attribute function returns an attribute value that
        is most frequent in the initial set, if the frequency is greater than 1. If not, all
        objects have different attribute values, and this method returns an empty set.
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
        Filters the set of objects, and returns those objects whose color is the most frequent
        color in the initial set of objects, if the highest frequency is greater than 1, or an
        empty set otherwise.

        This is an unusual name for what the method does, but just as ``blue`` filters objects to
        those that are blue, this filters objects to those that are of the same color.
        """
        return cls._get_objects_with_same_attribute(objects, lambda x: x.color)

    @classmethod
    def same_shape(cls, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent
        color in the initial set of objects, if the highest frequency is greater than 1, or an
        empty set otherwise.

        This is an unusual name for what the method does, but just as ``triangle`` filters objects
        to those that are triangles, this filters objects to those that are of the same shape.
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
    def top(cls, objects: Set[Object]) -> Set[Object]:
        """
        Return the topmost objects (i.e. maximum y_loc).
        """
        max_y_loc = max([obj.y_loc for obj in objects])
        return set([obj for obj in objects if obj.y_loc == max_y_loc])

    @classmethod
    def bottom(cls, objects: Set[Object]) -> Set[Object]:
        """
        Return the bottom most objects(i.e. minimum y_loc).
        """
        min_y_loc = min([obj.y_loc for obj in objects])
        return set([obj for obj in objects if obj.y_loc == min_y_loc])

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

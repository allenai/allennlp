"""
This module defines classes Object and Box (the two entities in the NLVR domain) and an NlvrWorld,
which mainly contains an execution method and related helper methods.
"""
from collections import defaultdict
import operator
from typing import List, Dict, Set, Callable, TypeVar, Union
import logging

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.semparse import util as semparse_util
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.type_declarations import nlvr_type_declaration as types
from allennlp.semparse.worlds.world import World

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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
    def __init__(self, attributes: JsonDict, box_id: str) -> None:
        object_color = attributes["color"].lower()
        # The dataset has a hex code only for blue for some reason.
        if object_color.startswith("#"):
            self.color = "blue"
        else:
            self.color = object_color
        object_shape = attributes["type"].lower()
        self.shape = object_shape
        self.x_loc = attributes["x_loc"]
        self.y_loc = attributes["y_loc"]
        self.size = attributes["size"]
        self._box_id = box_id

    def __str__(self):
        if self.size == 10:
            size = "small"
        elif self.size == 20:
            size = "medium"
        else:
            size = "big"
        return f"{size} {self.color} {self.shape} at ({self.x_loc}, {self.y_loc}) in {self._box_id}"

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
    box_id : ``int``
        An integer identifying the box index (0, 1 or 2).
    """
    def __init__(self,
                 objects_list: List[JsonDict],
                 box_id: int) -> None:
        self._name = f"box {box_id + 1}"
        self.objects = set([Object(object_dict, self._name) for object_dict in objects_list])

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

    # When we're converting from logical forms to action sequences, this set tells us which
    # functions in the logical form are curried functions, and how many arguments the function
    # actually takes.  This is necessary because NLTK curries all multi-argument functions to a
    # series of one-argument function applications.  See `world._get_transitions` for more info.
    curried_functions = {
            types.BOX_COLOR_FILTER_TYPE: 2,
            types.BOX_SHAPE_FILTER_TYPE: 2,
            types.BOX_COUNT_FILTER_TYPE: 2,
            types.ASSERT_COLOR_TYPE: 2,
            types.ASSERT_SHAPE_TYPE: 2,
            types.ASSERT_BOX_COUNT_TYPE: 2,
            types.ASSERT_OBJECT_COUNT_TYPE: 2,
            }

    # TODO(pradeep): Define more spatial relationship methods: left_of, right_of..
    # They should be defined for objects within the same box.
    def __init__(self, world_representation: List[List[JsonDict]]) -> None:
        super(NlvrWorld, self).__init__(global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                        global_name_mapping=types.COMMON_NAME_MAPPING,
                                        num_nested_lambdas=0)
        self._boxes = set([Box(object_list, box_id) for box_id, object_list in
                           enumerate(world_representation)])
        self._objects: Set[Object] = set()
        for box in self._boxes:
            self._objects.update(box.objects)

        self._number_operators = {"equals": operator.eq,
                                  "not_equals": operator.ne,
                                  "greater": operator.gt,
                                  "lesser": operator.lt,
                                  "greater_equals": operator.ge,
                                  "lesser_equals": operator.le}

        self._set_unary_operators = {"same": self._same,
                                     "different": self._different}

        self._set_binary_operators = {"all_equals": self._all_equals,
                                      "any_equals": self._any_equals,
                                      "none_equals": self._none_equals}

        self._count_functions = {"count": self._count,  # type: ignore
                                 "color_count": self._color_count,
                                 "shape_count": self._shape_count}

        self._attribute_functions = {"shape": self._shape,
                                     "color": self._color}

        # Mapping from terminal strings to productions that produce them.
        # Eg.: "yellow" -> "<o,o> -> yellow", "<b,<<b,e>,<e,b>>> -> filter_greater" etc.
        self.terminal_productions: Dict[str, str] = {}
        for constant in types.COMMON_NAME_MAPPING:
            alias = types.COMMON_NAME_MAPPING[constant]
            if alias in types.COMMON_TYPE_SIGNATURE:
                constant_type = types.COMMON_TYPE_SIGNATURE[alias]
                self.terminal_productions[constant] = "%s -> %s" % (constant_type, constant)

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return {types.TRUTH_TYPE}

    def _get_curried_functions(self) -> Dict[Type, int]:
        return NlvrWorld.curried_functions

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        return types.COMMON_NAME_MAPPING[name] if name in types.COMMON_NAME_MAPPING else name

    def get_agenda_for_sentence(self,
                                sentence: str,
                                add_paths_to_agenda: bool = False) -> List[str]:
        """
        Given a ``sentence``, returns a list of actions the sentence triggers as an ``agenda``. The
        ``agenda`` can be used while by a parser to guide the decoder.  sequences as possible. This
        is a simplistic mapping at this point, and can be expanded.

        Parameters
        ----------
        sentence : ``str``
            The sentence for which an agenda will be produced.
        add_paths_to_agenda : ``bool`` , optional
            If set, the agenda will also include nonterminal productions that lead to the terminals
            from the root node (default = False).
        """
        agenda = []
        sentence = sentence.lower()
        if sentence.startswith("there is a box") or sentence.startswith("there is a tower "):
            agenda.append(self.terminal_productions["box_exists"])
        elif sentence.startswith("there is a "):
            agenda.append(self.terminal_productions["object_exists"])

        if "<b,t> -> box_exists" not in agenda:
            # These are object filters and do not apply if we have a box_exists at the top.
            if "touch" in sentence:
                if "top" in sentence:
                    agenda.append(self.terminal_productions["touch_top"])
                elif "bottom" in sentence or "base" in sentence:
                    agenda.append(self.terminal_productions["touch_bottom"])
                elif "corner" in sentence:
                    agenda.append(self.terminal_productions["touch_corner"])
                elif "right" in sentence:
                    agenda.append(self.terminal_productions["touch_right"])
                elif "left" in sentence:
                    agenda.append(self.terminal_productions["touch_left"])
                elif "wall" in sentence or "edge" in sentence:
                    agenda.append(self.terminal_productions["touch_wall"])
                else:
                    agenda.append(self.terminal_productions["touch_object"])
            else:
                # The words "top" and "bottom" may be referring to top and bottom blocks in a tower.
                if "top" in sentence:
                    agenda.append(self.terminal_productions["top"])
                elif "bottom" in sentence or "base" in sentence:
                    agenda.append(self.terminal_productions["bottom"])

            if " not " in sentence:
                agenda.append(self.terminal_productions["negate_filter"])

        if " contains " in sentence or " has " in sentence:
            agenda.append(self.terminal_productions["all_boxes"])
        # This takes care of shapes, colors, top, bottom, big, small etc.
        for constant, production in self.terminal_productions.items():
            # TODO(pradeep): Deal with constant names with underscores.
            if "top" in constant or "bottom" in constant:
                # We already dealt with top, bottom, touch_top and touch_bottom above.
                continue
            if constant in sentence:
                if "<o,o> ->" in production and "<b,t> -> box_exists" in agenda:
                    if constant in ["square", "circle", "triangle"]:
                        agenda.append(self.terminal_productions[f"shape_{constant}"])
                    elif constant in ["yellow", "blue", "black"]:
                        agenda.append(self.terminal_productions[f"color_{constant}"])
                    else:
                        continue
                else:
                    agenda.append(production)
        # TODO (pradeep): Rules for "member_*" productions ("tower" or "box" followed by a color,
        # shape or number...)
        number_productions = self._get_number_productions(sentence)
        for production in number_productions:
            agenda.append(production)
        if not agenda:
            # None of the rules above was triggered!
            if "box" in sentence:
                agenda.append(self.terminal_productions["all_boxes"])
            else:
                agenda.append(self.terminal_productions["all_objects"])
        if add_paths_to_agenda:
            agenda = self._add_nonterminal_productions(agenda)
        return agenda

    @staticmethod
    def _get_number_productions(sentence: str) -> List[str]:
        """
        Gathers all the numbers in the sentence, and returns productions that lead to them.
        """
        # The mapping here is very simple and limited, which also shouldn't be a problem
        # because numbers seem to be represented fairly regularly.
        number_strings = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six":
                          "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        number_productions = []
        tokens = sentence.split()
        numbers = number_strings.values()
        for token in tokens:
            if token in numbers:
                number_productions.append(f"e -> {token}")
            elif token in number_strings:
                number_productions.append(f"e -> {number_strings[token]}")
        return number_productions

    def _add_nonterminal_productions(self, agenda: List[str]) -> List[str]:
        """
        Given a partially populated agenda with (mostly) terminal productions, this method adds the
        nonterminal productions that lead from the root to the terminal productions.
        """
        nonterminal_productions = set(agenda)
        for action in agenda:
            paths = self.get_paths_to_root(action, max_num_paths=5)
            for path in paths:
                for path_action in path:
                    nonterminal_productions.add(path_action)
        new_agenda = list(nonterminal_productions)
        return new_agenda

    ## Complex operators
    @staticmethod
    def _same(input_set: Set[str]) -> bool:
        return len(input_set) == 1

    @staticmethod
    def _different(input_set: Set[str]) -> bool:
        return len(input_set) != 1

    @staticmethod
    def _all_equals(input_set: Set[str], target_value) -> bool:
        if not input_set:
            return False
        return all([x == target_value for x in input_set])

    @staticmethod
    def _any_equals(input_set: Set[str], target_value) -> bool:
        return any([x == target_value for x in input_set])

    ## Attribute functions
    @staticmethod
    def _color(objects: Set[Object]) -> Set[str]:
        """
        Returns the set of colors of a set of objects.
        """
        return set([obj.color for obj in objects])

    @staticmethod
    def _shape(objects: Set[Object]) -> Set[str]:
        """
        Returns the set of shapes of a set of objects.
        """
        return set([obj.shape for obj in objects])

    @staticmethod
    def _count(entities_set: Union[Set[Box], Set[Object]]) -> int:
        return len(entities_set)

    @staticmethod
    def _none_equals(input_set: Set[str], target_value) -> bool:
        return all([x != target_value for x in input_set])

    @classmethod
    def _shape_count(cls, input_set: Set[Object]) -> int:
        return len(cls._shape(input_set))

    @classmethod
    def _color_count(cls, input_set: Set[Object]) -> int:
        return len(cls._color(input_set))

    def execute(self, logical_form: str) -> bool:
        """
        Execute the logical form. The top level function is an assertion function (see below). We
        just parse the string into a list and pass the whole thing to ``_execute_assertion`` and let
        the method deal with it. This is because the dataset contains sentences (instead of
        questions), and they evaluate to either true or false.

        The language we defined here contains six types of functions, five of which return sets,
        and one returns booleans.

        1) Assertion Function : These occur only at the root node of the logical form trees. They
        take a set of entities, and compare their attributes to a given value, and return true or
        false. The entities they take can be boxes or objects. If the assertion function takes
        objects, it may compare their colors or shapes with the given value; If it takes boxes,
        the attributes it compares are only the counts. The comparison operator can be any of
        equals, not equals, greater than, etc. So, the function specifies what kind of entities it
        takes, the attribute being compared and the comparison operator. For example,
        "object_count_not_equals" takes a set of objects, compares their count to the given value
        and returns true iff they are not equal. They have names like "object_*" or "box_*"

        2) Object Attribute Functions: They take sets of objects and return sets of attributes.
        `color` and `shape` are the attribute functions.

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

    # TODO(pradeep): The methods ``_execute_assertion``, ``_execute_box_filter`` and
    # ``execute_object_filter`` are very complex a this point. I should break these down into a
    # simpler set of functions, such that one there is a method for each terminal, and the
    # "execution logic" is minimal.

    def _execute_assertion(self, sub_expression: List) -> bool:
        """
        Assertion functions are boolean functions. They are of two types:
        1) Exists functions: They take one argument, a set and check whether it is not empty.
        Syntax: ``(exists_function function_returning_entities)``
        Example: ``(object_exists (black (top all_objects)))`` ("There is a black object at the top
        of a tower.")
        2) Other assert functions: They take two arguments, which evaluate to strings or integers,
        and compare them. The first element in the input list should be the assertion function name,
        the second a function returning entities, and the last element should be a constant. The
        assertion function should specify the entity type, the attribute being compared, and a
        comparison operator, in that order separated by underscores. The following are the expected
        values:
            Entity types: ``object``, ``box``
            Attributes being compared: ``color``, ``shape``, ``count``, ``color_count``,
            ``shape_count``
            Comparison operator:
                Applicable to sets: ``all_equals``, ``any_equals``, ``none_equals``, ``same``,
                ``different``
                Applicable to counts: ``equals``, ``not_equals``, ``lesser``, ``lesser_equals``,
                ``greater``, ``greater_equals``
        Syntax: ``(assertion_function function_returning_entities constant)``
        Example: ``(box_count_equals (member_shape_equals all_boxes shape_square) 2)``
        ("There are exactly two boxes with only squares in them")

        Note that the first kind is a special case of the second where the attribute type is
        ``count``, comparison operator is ``greater_equals`` and the constant is ``1``.
        """
        # TODO(pradeep): We may want to change the order of arguments here to make decoding easier.
        assert isinstance(sub_expression, list), "Invalid assertion expression: %s" % sub_expression
        if len(sub_expression) == 1 and isinstance(sub_expression[0], list):
            return self._execute_assertion(sub_expression[0])
        is_assert_function = sub_expression[0].startswith("object_") or \
        sub_expression[0].startswith("box_")
        assert isinstance(sub_expression[0], str) and is_assert_function,\
               "Invalid assertion function: %s" % (sub_expression[0])
        # Example: box_count_not_equals, entities being evaluated are boxes, the relevant attibute
        # is their count, and the function will return true if the attribute is not equal to the
        # target.
        function_name_parts = sub_expression[0].split('_')
        entity_type = function_name_parts[0]
        target_attribute = None
        if len(function_name_parts) == 2 and function_name_parts[1] == "exists":
            attribute_type = "count"
            comparison_op = "greater_equals"
            target_attribute = 1
        else:
            target_attribute = self._execute_constant(sub_expression[2])
            # If the length of ``function_name_parts`` is 3, getting the attribute and comparison
            # operator is easy. However, if it is greater than 3, we need to determine where the
            # attribute function stops and where the comparison operator begins.
            if len(function_name_parts) == 3:
                # These are cases like ``object_color_equals``, ``box_count_greater`` etc.
                attribute_type = function_name_parts[1]
                comparison_op = function_name_parts[2]
            elif function_name_parts[2] == 'count':
                # These are cases like ``object_color_count_equals``,
                # ``object_shape_count_greater_equals`` etc.
                attribute_type = "_".join(function_name_parts[1:3])
                comparison_op = "_".join(function_name_parts[3:])
            else:
                # These are cases like ``box_count_greater_equals``, ``object_shape_not_equals``
                # etc.
                attribute_type = function_name_parts[1]
                comparison_op = "_".join(function_name_parts[2:])

        entity_expression = sub_expression[1]
        returned_count = None
        returned_attribute = None
        if entity_type == "box":
            # You can only count boxes. The other attributes do not apply.
            returned_count = self._count(self._execute_box_filter(entity_expression))
        elif "count" in attribute_type:
            # We're counting objects, colors or shapes.
            count_function = self._count_functions[attribute_type]
            returned_count = count_function(self._execute_object_filter(entity_expression))
        else:
            # We're getting colors or shapes from objects.
            attribute_function = self._attribute_functions[attribute_type]
            returned_attribute = attribute_function(self._execute_object_filter(entity_expression))

        if comparison_op in ["all_equals", "any_equals", "none_equals"]:
            set_comparison = self._set_binary_operators[comparison_op]
            if returned_attribute is None:
                logger.error("Invalid assertion function: %s", sub_expression[0])
                raise ExecutionError("Invalid assertion function")
            return set_comparison(returned_attribute, target_attribute)
        else:
            number_comparison = self._number_operators[comparison_op]
            if returned_count is None:
                logger.error("Invalid assertion function: %s", sub_expression[0])
                raise ExecutionError("Invalid assertion function")
            return number_comparison(returned_count, target_attribute)

    def _execute_box_filter(self, sub_expression: Union[str, List]) -> Set[Box]:
        """
        Box filtering functions either apply a filter on a set of boxes and return the filtered set,
        or return all the boxes.
        The elements should evaluate to one of the following:
        ``(box_filtering_function set_to_filter constant)`` or
        ``all_boxes``

        In the first kind of forms, the ``box_filtering_function`` also specifies the attribute
        being compared and the comparison operator. The attribute is of the objects contained in
        each box in the ``set_to_filter``.
        Example: ``(member_color_count_greater all_boxes 1)``
        filters all boxes by extracting the colors of the objects in each of them, and returns a
        subset of boxes from the original set where the number of colors of objects is greater than
        1.
        """
        # TODO(pradeep): We may want to change the order of arguments here to make decoding easier.
        if sub_expression[0].startswith('member_'):
            function_name_parts = sub_expression[0].split("_")
            if len(function_name_parts) == 3:
                attribute_type = function_name_parts[1]
                comparison_op = function_name_parts[2]
            elif function_name_parts[2] == "count":
                attribute_type = "_".join(function_name_parts[1:3])
                comparison_op = "_".join(function_name_parts[3:])
            else:
                attribute_type = function_name_parts[1]
                comparison_op = "_".join(function_name_parts[2:])
            set_to_filter = self._execute_box_filter(sub_expression[1])
            return_set = set()
            if comparison_op in ["same", "different"]:
                # We don't need a target attribute for these functions, and the "comparison" is done
                # on sets.
                comparison_function = self._set_unary_operators[comparison_op]
                for box in set_to_filter:
                    returned_attribute: Set[str] = self._attribute_functions[attribute_type](box.objects)
                    if comparison_function(returned_attribute):
                        return_set.add(box)
            else:
                target_attribute = self._execute_constant(sub_expression[-1])
                is_set_operation = comparison_op in ["all_equals", "any_equals", "none_equals"]
                # These are comparisons like equals, greater etc, and we need a target attribute
                # which we first evaluate here. Then, the returned attribute (if it is a singleton
                # set or an integer), is compared against the target attribute.
                for box in set_to_filter:
                    if is_set_operation:
                        returned_attribute = self._attribute_functions[attribute_type](box.objects)
                        box_wanted = self._set_binary_operators[comparison_op](returned_attribute,
                                                                               target_attribute)
                    else:
                        returned_count = self._count_functions[attribute_type](box.objects)
                        box_wanted = self._number_operators[comparison_op](returned_count,
                                                                           target_attribute)
                    if box_wanted:
                        return_set.add(box)
            return return_set
        elif sub_expression == 'all_boxes' or sub_expression[0] == 'all_boxes':
            return self._boxes
        else:
            logger.error("Invalid box filter expression: %s", sub_expression)
            raise ExecutionError("Unknown box filter expression")

    def _execute_object_filter(self, sub_expression: Union[str, List]) -> Set[Object]:
        """
        Object filtering functions should either be a string referring to all objects, or list which
        executes to a filtering operation.
        The elements should evaluate to one of the following:
            (object_filtering_function object_set)
            ((negate_filter object_filtering_function) object_set)
            all_objects
        """
        if sub_expression[0][0] == "negate_filter":
            initial_set = self._execute_object_filter(sub_expression[1])
            original_filter_name = sub_expression[0][1]
            # It is possible that the decoder has produced a sequence of nested negations. We deal
            # with that here.
            # TODO (pradeep): This is messy. Fix the type declaration so that we don't have to deal
            # with this.
            num_negations = 1
            while isinstance(original_filter_name, list) and \
                  original_filter_name[0] == "negate_filter":
                # We have a sequence of "negate_filters"
                num_negations += 1
                original_filter_name = original_filter_name[1]
            if num_negations % 2 == 0:
                return initial_set
            try:
                original_filter = getattr(self, original_filter_name)
                return self.negate_filter(original_filter, initial_set)
            except AttributeError:
                logger.error("Function not found: %s", original_filter_name)
                raise ExecutionError("Function not found")
        elif sub_expression == "all_objects" or sub_expression[0] == "all_objects":
            return self._objects
        elif isinstance(sub_expression[0], str) and len(sub_expression) == 2:
            # These are functions like black, square, same_color etc.
            function = None
            try:
                function = getattr(self, sub_expression[0])
            except AttributeError:
                logger.error("Function not found: %s", sub_expression[0])
                raise ExecutionError("Function not found")
            arguments = sub_expression[1]
            if isinstance(arguments, list) and str(arguments[0]).startswith("member_") or \
                arguments == 'all_boxes' or arguments[0] == 'all_boxes':
                if sub_expression[0] != "object_in_box":
                    logger.error("Invalid object filter expression: %s", sub_expression)
                    raise ExecutionError("Invalid object filter expression")
                return function(self._execute_box_filter(arguments))
            else:
                return function(self._execute_object_filter(arguments))
        else:
            logger.error("Invalid object filter expression: %s", sub_expression)
            raise ExecutionError("Invalid object filter expression")

    @staticmethod
    def _execute_constant(sub_expression: str):
        """
        Acceptable constants are numbers or strings starting with `shape_` or `color_`
        """
        if not isinstance(sub_expression, str):
            logger.error("Invalid constant: %s", sub_expression)
            raise ExecutionError("Invalid constant")
        if str.isdigit(sub_expression):
            return int(sub_expression)
        elif sub_expression.startswith('color_'):
            return sub_expression.replace('color_', '')
        elif sub_expression.startswith('shape_'):
            return sub_expression.replace('shape_', '')
        else:
            logger.error("Invalid constant: %s", sub_expression)
            raise ExecutionError("Invalid constant")

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

    @staticmethod
    def object_in_box(box: Set[Box]) -> Set[Object]:
        return_set: Set[Object] = set()
        for box_ in box:
            return_set.update(box_.objects)
        return return_set

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
        if not objects_of_attribute:
            return set()
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
        return set([obj for obj in objects if obj.y_loc + obj.size == 100])

    @classmethod
    def touch_left(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.x_loc == 0])

    @classmethod
    def touch_top(cls, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.y_loc == 0])

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

    def touch_object(self, objects: Set[Object]) -> Set[Object]:
        """
        Returns all objects that touch the given set of objects.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set = set()
        for box, box_objects in objects_per_box.items():
            candidate_objects = box.objects
            for object_ in box_objects:
                for candidate_object in candidate_objects:
                    if self._objects_touch_each_other(object_, candidate_object):
                        return_set.add(candidate_object)
        return return_set

    @classmethod
    def _objects_touch_each_other(cls, object1: Object, object2: Object) -> bool:
        """
        Returns true iff the objects touch each other.
        """
        in_vertical_range = object1.y_loc <= object2.y_loc + object2.size and \
                            object1.y_loc + object1.size >= object2.y_loc
        in_horizantal_range = object1.x_loc <= object2.x_loc + object2.size and \
                            object1.x_loc + object1.size >= object2.x_loc
        touch_side = object1.x_loc + object1.size == object2.x_loc or \
                     object2.x_loc + object2.size == object1.x_loc
        touch_top_or_bottom = object1.y_loc + object1.size == object2.y_loc or \
                              object2.y_loc + object2.size == object1.y_loc
        return (in_vertical_range and touch_side) or (in_horizantal_range and touch_top_or_bottom)

    def top(self, objects: Set[Object]) -> Set[Object]:
        """
        Return the topmost objects (i.e. minimum y_loc). The comparison is done separately for each
        box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set: Set[Object] = set()
        for _, box_objects in objects_per_box.items():
            min_y_loc = min([obj.y_loc for obj in box_objects])
            return_set.update(set([obj for obj in box_objects if obj.y_loc == min_y_loc]))
        return return_set

    def bottom(self, objects: Set[Object]) -> Set[Object]:
        """
        Return the bottom most objects(i.e. maximum y_loc). The comparison is done separately for
        each box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set: Set[Object] = set()
        for _, box_objects in objects_per_box.items():
            max_y_loc = max([obj.y_loc for obj in box_objects])
            return_set.update(set([obj for obj in box_objects if obj.y_loc == max_y_loc]))
        return return_set

    def above(self, objects: Set[Object]) -> Set[Object]:
        """
        Returns the set of objects in the same boxes that are above the given objects. That is, if
        the input is a set of two objects, one in each box, we will return a union of the objects
        above the first object in the first box, and those above the second object in the second box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set = set()
        for box in objects_per_box:
            # min_y_loc corresponds to the top-most object.
            min_y_loc = min([obj.y_loc for obj in objects_per_box[box]])
            for candidate_obj in box.objects:
                if candidate_obj.y_loc < min_y_loc:
                    return_set.add(candidate_obj)
        return return_set

    def below(self, objects: Set[Object]) -> Set[Object]:
        """
        Returns the set of objects in the same boxes that are below the given objects. That is, if
        the input is a set of two objects, one in each box, we will return a union of the objects
        below the first object in the first box, and those below the second object in the second box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set = set()
        for box in objects_per_box:
            # max_y_loc corresponds to the bottom-most object.
            max_y_loc = max([obj.y_loc for obj in objects_per_box[box]])
            for candidate_obj in box.objects:
                if candidate_obj.y_loc > max_y_loc:
                    return_set.add(candidate_obj)
        return return_set

    def _separate_objects_by_boxes(self, objects: Set[Object]) -> Dict[Box, List[Object]]:
        """
        Given a set of objects, separate them by the boxes they belong to and return a dict.
        """
        objects_per_box: Dict[Box, List[Object]] = defaultdict(list)
        for box in self._boxes:
            for object_ in objects:
                if object_ in box.objects:
                    objects_per_box[box].append(object_)
        return objects_per_box

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

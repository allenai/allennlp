from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Set

from allennlp.common.util import JsonDict
from allennlp.semparse.domain_languages.domain_language import DomainLanguage, predicate


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
        self._objects_string = str([str(_object) for _object in objects_list])
        self.objects = set([Object(object_dict, self._name) for object_dict in objects_list])
        self.colors = set([obj.color for obj in self.objects])
        self.shapes = set([obj.shape for obj in self.objects])

    def __str__(self):
        return self._objects_string

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Color(NamedTuple):
    color: str


class Shape(NamedTuple):
    shape: str


class NlvrLanguage(DomainLanguage):
    # pylint: disable=no-self-use,too-many-public-methods
    def __init__(self, boxes: Set[Box]) -> None:
        self.boxes = boxes
        self.objects: Set[Object] = set()
        for box in self.boxes:
            self.objects.update(box.objects)
        allowed_constants = {
                'color_blue': Color('blue'),
                'color_black': Color('black'),
                'color_yellow': Color('yellow'),
                'shape_triangle': Shape('triangle'),
                'shape_square': Shape('square'),
                'shape_circle': Shape('circle'),
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                }
        super().__init__(start_types={bool}, allowed_constants=allowed_constants)

        # Mapping from terminal strings to productions that produce them.
        # Eg.: "yellow" -> "<Set[Object]:Set[Object]> -> yellow"
        # We use this in the agenda-related methods, and some models that use this language look at
        # this field to know how many terminals to plan for.
        self.terminal_productions: Dict[str, str] = {}
        for name, types in self._function_types.items():
            self.terminal_productions[name] = f"{types[0]} -> {name}"

    # These first two methods are about getting an "agenda", which, given an input utterance,
    # tries to guess what production rules should be needed in the logical form.

    def get_agenda_for_sentence(self, sentence: str) -> List[str]:
        """
        Given a ``sentence``, returns a list of actions the sentence triggers as an ``agenda``. The
        ``agenda`` can be used while by a parser to guide the decoder.  sequences as possible. This
        is a simplistic mapping at this point, and can be expanded.

        Parameters
        ----------
        sentence : ``str``
            The sentence for which an agenda will be produced.
        """
        agenda = []
        sentence = sentence.lower()
        if sentence.startswith("there is a box") or sentence.startswith("there is a tower "):
            agenda.append(self.terminal_productions["box_exists"])
        elif sentence.startswith("there is a "):
            agenda.append(self.terminal_productions["object_exists"])

        if "<Set[Box]:bool> -> box_exists" not in agenda:
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
                if "<Set[Object]:Set[Object]> ->" in production and "<Set[Box]:bool> -> box_exists" in agenda:
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
                number_productions.append(f"int -> {token}")
            elif token in number_strings:
                number_productions.append(f"int -> {number_strings[token]}")
        return number_productions

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.boxes == other.boxes and self.objects == other.objects
        return NotImplemented

    # All methods below here are predicates in the NLVR language, or helper methods for them.

    @predicate
    def all_boxes(self) -> Set[Box]:
        return self.boxes

    @predicate
    def all_objects(self) -> Set[Object]:
        return self.objects

    @predicate
    def box_exists(self, boxes: Set[Box]) -> bool:
        return len(boxes) > 0

    @predicate
    def object_exists(self, objects: Set[Object]) -> bool:
        return len(objects) > 0

    @predicate
    def object_in_box(self, box: Set[Box]) -> Set[Object]:
        return_set: Set[Object] = set()
        for box_ in box:
            return_set.update(box_.objects)
        return return_set

    @predicate
    def black(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.color == "black"])

    @predicate
    def blue(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.color == "blue"])

    @predicate
    def yellow(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.color == "yellow"])

    @predicate
    def circle(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.shape == "circle"])

    @predicate
    def square(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.shape == "square"])

    @predicate
    def triangle(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.shape == "triangle"])

    @predicate
    def same_color(self, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent
        color in the initial set of objects, if the highest frequency is greater than 1, or an
        empty set otherwise.

        This is an unusual name for what the method does, but just as ``blue`` filters objects to
        those that are blue, this filters objects to those that are of the same color.
        """
        return self._get_objects_with_same_attribute(objects, lambda x: x.color)

    @predicate
    def same_shape(self, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent
        color in the initial set of objects, if the highest frequency is greater than 1, or an
        empty set otherwise.

        This is an unusual name for what the method does, but just as ``triangle`` filters objects
        to those that are triangles, this filters objects to those that are of the same shape.
        """
        return self._get_objects_with_same_attribute(objects, lambda x: x.shape)

    @predicate
    def touch_bottom(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.y_loc + obj.size == 100])

    @predicate
    def touch_left(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.x_loc == 0])

    @predicate
    def touch_top(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.y_loc == 0])

    @predicate
    def touch_right(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.x_loc + obj.size == 100])

    @predicate
    def touch_wall(self, objects: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(self.touch_top(objects), self.touch_left(objects),
                                self.touch_right(objects), self.touch_bottom(objects))

    @predicate
    def touch_corner(self, objects: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(self.touch_top(objects).intersection(self.touch_right(objects)),
                                self.touch_top(objects).intersection(self.touch_left(objects)),
                                self.touch_bottom(objects).intersection(self.touch_right(objects)),
                                self.touch_bottom(objects).intersection(self.touch_left(objects)))

    @predicate
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

    @predicate
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

    @predicate
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

    @predicate
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

    @predicate
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

    @predicate
    def small(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.size == 10])

    @predicate
    def medium(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.size == 20])

    @predicate
    def big(self, objects: Set[Object]) -> Set[Object]:
        return set([obj for obj in objects if obj.size == 30])

    @predicate
    def box_count_equals(self, boxes: Set[Box], count: int) -> bool:
        return len(boxes) == count

    @predicate
    def box_count_not_equals(self, boxes: Set[Box], count: int) -> bool:
        return len(boxes) != count

    @predicate
    def box_count_greater(self, boxes: Set[Box], count: int) -> bool:
        return len(boxes) > count

    @predicate
    def box_count_greater_equals(self, boxes: Set[Box], count: int) -> bool:
        return len(boxes) >= count

    @predicate
    def box_count_lesser(self, boxes: Set[Box], count: int) -> bool:
        return len(boxes) < count

    @predicate
    def box_count_lesser_equals(self, boxes: Set[Box], count: int) -> bool:
        return len(boxes) <= count

    @predicate
    def object_color_all_equals(self, objects: Set[Object], color: Color) -> bool:
        return all([obj.color == color.color for obj in objects])

    @predicate
    def object_color_any_equals(self, objects: Set[Object], color: Color) -> bool:
        return any([obj.color == color.color for obj in objects])

    @predicate
    def object_color_none_equals(self, objects: Set[Object], color: Color) -> bool:
        return all([obj.color != color.color for obj in objects])

    @predicate
    def object_shape_all_equals(self, objects: Set[Object], shape: Shape) -> bool:
        return all([obj.shape == shape.shape for obj in objects])

    @predicate
    def object_shape_any_equals(self, objects: Set[Object], shape: Shape) -> bool:
        return any([obj.shape == shape.shape for obj in objects])

    @predicate
    def object_shape_none_equals(self, objects: Set[Object], shape: Shape) -> bool:
        return all([obj.shape != shape.shape for obj in objects])

    @predicate
    def object_count_equals(self, objects: Set[Object], count: int) -> bool:
        return len(objects) == count

    @predicate
    def object_count_not_equals(self, objects: Set[Object], count: int) -> bool:
        return len(objects) != count

    @predicate
    def object_count_greater(self, objects: Set[Object], count: int) -> bool:
        return len(objects) > count

    @predicate
    def object_count_greater_equals(self, objects: Set[Object], count: int) -> bool:
        return len(objects) >= count

    @predicate
    def object_count_lesser(self, objects: Set[Object], count: int) -> bool:
        return len(objects) < count

    @predicate
    def object_count_lesser_equals(self, objects: Set[Object], count: int) -> bool:
        return len(objects) <= count

    @predicate
    def object_color_count_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.color for obj in objects])) == count

    @predicate
    def object_color_count_not_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.color for obj in objects])) != count

    @predicate
    def object_color_count_greater(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.color for obj in objects])) > count

    @predicate
    def object_color_count_greater_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.color for obj in objects])) >= count

    @predicate
    def object_color_count_lesser(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.color for obj in objects])) < count

    @predicate
    def object_color_count_lesser_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.color for obj in objects])) <= count

    @predicate
    def object_shape_count_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.shape for obj in objects])) == count

    @predicate
    def object_shape_count_not_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.shape for obj in objects])) != count

    @predicate
    def object_shape_count_greater(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.shape for obj in objects])) > count

    @predicate
    def object_shape_count_greater_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.shape for obj in objects])) >= count

    @predicate
    def object_shape_count_lesser(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.shape for obj in objects])) < count

    @predicate
    def object_shape_count_lesser_equals(self, objects: Set[Object], count: int) -> bool:
        return len(set([obj.shape for obj in objects])) <= count

    @predicate
    def member_count_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.objects) == count])

    @predicate
    def member_count_not_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.objects) != count])

    @predicate
    def member_count_greater(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.objects) > count])

    @predicate
    def member_count_greater_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.objects) >= count])

    @predicate
    def member_count_lesser(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.objects) < count])

    @predicate
    def member_count_lesser_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.objects) <= count])

    @predicate
    def member_color_count_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.colors) == count])

    @predicate
    def member_color_count_not_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.colors) != count])

    @predicate
    def member_color_count_greater(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.colors) > count])

    @predicate
    def member_color_count_greater_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.colors) >= count])

    @predicate
    def member_color_count_lesser(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.colors) < count])

    @predicate
    def member_color_count_lesser_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.colors) <= count])

    @predicate
    def member_shape_count_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.shapes) == count])

    @predicate
    def member_shape_count_not_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.shapes) != count])

    @predicate
    def member_shape_count_greater(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.shapes) > count])

    @predicate
    def member_shape_count_greater_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.shapes) >= count])

    @predicate
    def member_shape_count_lesser(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.shapes) < count])

    @predicate
    def member_shape_count_lesser_equals(self, boxes: Set[Box], count: int) -> Set[Box]:
        return set([box for box in boxes if len(box.shapes) <= count])

    @predicate
    def member_color_all_equals(self, boxes: Set[Box], color: Color) -> Set[Box]:
        return set([box for box in boxes if self.object_color_all_equals(box.objects, color)])

    @predicate
    def member_color_any_equals(self, boxes: Set[Box], color: Color) -> Set[Box]:
        return set([box for box in boxes if self.object_color_any_equals(box.objects, color)])

    @predicate
    def member_color_none_equals(self, boxes: Set[Box], color: Color) -> Set[Box]:
        return set([box for box in boxes if self.object_color_none_equals(box.objects, color)])

    @predicate
    def member_shape_all_equals(self, boxes: Set[Box], shape: Shape) -> Set[Box]:
        return set([box for box in boxes if self.object_shape_all_equals(box.objects, shape)])

    @predicate
    def member_shape_any_equals(self, boxes: Set[Box], shape: Shape) -> Set[Box]:
        return set([box for box in boxes if self.object_shape_any_equals(box.objects, shape)])

    @predicate
    def member_shape_none_equals(self, boxes: Set[Box], shape: Shape) -> Set[Box]:
        return set([box for box in boxes if self.object_shape_none_equals(box.objects, shape)])

    @predicate
    def member_shape_same(self, boxes: Set[Box]) -> Set[Box]:
        return set([box for box in boxes if self.object_shape_count_equals(box.objects, 1)])

    @predicate
    def member_color_same(self, boxes: Set[Box]) -> Set[Box]:
        return set([box for box in boxes if self.object_color_count_equals(box.objects, 1)])

    @predicate
    def member_shape_different(self, boxes: Set[Box]) -> Set[Box]:
        return set([box for box in boxes if self.object_shape_count_not_equals(box.objects, 1)])

    @predicate
    def member_color_different(self, boxes: Set[Box]) -> Set[Box]:
        return set([box for box in boxes if self.object_color_count_not_equals(box.objects, 1)])

    @predicate
    def negate_filter(self, filter_function: Callable[[Set[Object]], Set[Object]]) -> Callable[[Set[Object]],
                                                                                               Set[Object]]:
        def negated_filter(objects: Set[Object]) -> Set[Object]:
            return objects.difference(filter_function(objects))
        return negated_filter

    def _objects_touch_each_other(self, object1: Object, object2: Object) -> bool:
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

    def _separate_objects_by_boxes(self, objects: Set[Object]) -> Dict[Box, List[Object]]:
        """
        Given a set of objects, separate them by the boxes they belong to and return a dict.
        """
        objects_per_box: Dict[Box, List[Object]] = defaultdict(list)
        for box in self.boxes:
            for object_ in objects:
                if object_ in box.objects:
                    objects_per_box[box].append(object_)
        return objects_per_box

    def _get_objects_with_same_attribute(self,
                                         objects: Set[Object],
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

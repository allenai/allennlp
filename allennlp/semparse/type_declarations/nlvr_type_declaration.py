from typing import List, Set
from overrides import overrides

from nltk.sem.logic import TRUTH_TYPE, BasicType, EntityType, Type

from allennlp.semparse.type_declarations.type_declaration import (ComplexType, HigherOrderType,
                                                                  NamedBasicType, NameMapper)


class NegateFilterType(HigherOrderType):
    """
    Because our negate filters are higher-order functions, we need to make an explicit class here,
    to make sure that we've overridden the right methods correctly.
    """
    def __init__(self, first, second):
        super().__init__(num_arguments=1, first=first, second=second)

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        # There's no ANY_TYPE in here, so we don't need to do any substitution.
        return [self]


# All constants default to ``EntityType`` in NLTK. For domains where constants of different types
# appear in the logical forms, we have a way of specifying ``constant_type_prefixes`` and passing
# them to the constructor of ``World``. However, in the NLVR language we defined, we see constants
# of just one type, number. So we let them default to ``EntityType``.
NUM_TYPE = EntityType()
BOX_TYPE = NamedBasicType("BOX")
OBJECT_TYPE = NamedBasicType("OBJECT")
COLOR_TYPE = NamedBasicType("COLOR")
SHAPE_TYPE = NamedBasicType("SHAPE")
OBJECT_FILTER_TYPE = ComplexType(OBJECT_TYPE, OBJECT_TYPE)
NEGATE_FILTER_TYPE = NegateFilterType(ComplexType(OBJECT_TYPE, OBJECT_TYPE),
                                      ComplexType(OBJECT_TYPE, OBJECT_TYPE))
BOX_MEMBERSHIP_TYPE = ComplexType(BOX_TYPE, OBJECT_TYPE)

BOX_COLOR_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(COLOR_TYPE, BOX_TYPE))
BOX_SHAPE_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(SHAPE_TYPE, BOX_TYPE))
BOX_COUNT_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(NUM_TYPE, BOX_TYPE))
# This box filter returns boxes where a specified attribute is same or different
BOX_ATTRIBUTE_SAME_FILTER_TYPE = ComplexType(BOX_TYPE, BOX_TYPE)


ASSERT_COLOR_TYPE = ComplexType(OBJECT_TYPE, ComplexType(COLOR_TYPE, TRUTH_TYPE))
ASSERT_SHAPE_TYPE = ComplexType(OBJECT_TYPE, ComplexType(SHAPE_TYPE, TRUTH_TYPE))
ASSERT_BOX_COUNT_TYPE = ComplexType(BOX_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))
ASSERT_OBJECT_COUNT_TYPE = ComplexType(OBJECT_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))

BOX_EXISTS_TYPE = ComplexType(BOX_TYPE, TRUTH_TYPE)
OBJECT_EXISTS_TYPE = ComplexType(OBJECT_TYPE, TRUTH_TYPE)

BASIC_TYPES = {NUM_TYPE, BOX_TYPE, OBJECT_TYPE, COLOR_TYPE, SHAPE_TYPE}

name_mapper = NameMapper()  # pylint: disable=invalid-name

# Entities
name_mapper.map_name_with_signature("all_objects", OBJECT_TYPE)
name_mapper.map_name_with_signature("all_boxes", BOX_TYPE)
name_mapper.map_name_with_signature("color_black", COLOR_TYPE)
name_mapper.map_name_with_signature("color_blue", COLOR_TYPE)
name_mapper.map_name_with_signature("color_yellow", COLOR_TYPE)
name_mapper.map_name_with_signature("shape_triangle", SHAPE_TYPE)
name_mapper.map_name_with_signature("shape_square", SHAPE_TYPE)
name_mapper.map_name_with_signature("shape_circle", SHAPE_TYPE)


# Attribute function
name_mapper.map_name_with_signature("object_in_box", BOX_MEMBERSHIP_TYPE)


# Assert functions
name_mapper.map_name_with_signature("object_color_all_equals", ASSERT_COLOR_TYPE)
name_mapper.map_name_with_signature("object_color_any_equals", ASSERT_COLOR_TYPE)
name_mapper.map_name_with_signature("object_color_none_equals", ASSERT_COLOR_TYPE)
name_mapper.map_name_with_signature("object_shape_all_equals", ASSERT_SHAPE_TYPE)
name_mapper.map_name_with_signature("object_shape_any_equals", ASSERT_SHAPE_TYPE)
name_mapper.map_name_with_signature("object_shape_none_equals", ASSERT_SHAPE_TYPE)
name_mapper.map_name_with_signature("box_count_equals", ASSERT_BOX_COUNT_TYPE)
name_mapper.map_name_with_signature("box_count_not_equals", ASSERT_BOX_COUNT_TYPE)
name_mapper.map_name_with_signature("box_count_greater", ASSERT_BOX_COUNT_TYPE)
name_mapper.map_name_with_signature("box_count_greater_equals", ASSERT_BOX_COUNT_TYPE)
name_mapper.map_name_with_signature("box_count_lesser", ASSERT_BOX_COUNT_TYPE)
name_mapper.map_name_with_signature("box_count_lesser_equals", ASSERT_BOX_COUNT_TYPE)
name_mapper.map_name_with_signature("object_count_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_count_not_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_count_greater", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_count_greater_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_count_lesser", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_count_lesser_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_color_count_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_color_count_not_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_color_count_greater", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_color_count_greater_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_color_count_lesser", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_color_count_lesser_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_shape_count_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_shape_count_not_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_shape_count_greater", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_shape_count_greater_equals", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_shape_count_lesser", ASSERT_OBJECT_COUNT_TYPE)
name_mapper.map_name_with_signature("object_shape_count_lesser_equals", ASSERT_OBJECT_COUNT_TYPE)

name_mapper.map_name_with_signature("box_exists", BOX_EXISTS_TYPE)
name_mapper.map_name_with_signature("object_exists", OBJECT_EXISTS_TYPE)


# Box filter functions
name_mapper.map_name_with_signature("member_count_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_count_not_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_all_equals", BOX_SHAPE_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_any_equals", BOX_SHAPE_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_none_equals", BOX_SHAPE_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_all_equals", BOX_COLOR_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_any_equals", BOX_COLOR_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_none_equals", BOX_COLOR_FILTER_TYPE)
name_mapper.map_name_with_signature("member_count_greater", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_count_greater_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_count_lesser", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_count_lesser_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_count_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_count_not_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_count_greater", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_count_greater_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_count_lesser", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_count_lesser_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_count_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_count_not_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_count_greater", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_count_greater_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_count_lesser", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_count_lesser_equals", BOX_COUNT_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_same", BOX_ATTRIBUTE_SAME_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_same", BOX_ATTRIBUTE_SAME_FILTER_TYPE)
name_mapper.map_name_with_signature("member_shape_different", BOX_ATTRIBUTE_SAME_FILTER_TYPE)
name_mapper.map_name_with_signature("member_color_different", BOX_ATTRIBUTE_SAME_FILTER_TYPE)


# Object filter functions
name_mapper.map_name_with_signature("black", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("blue", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("yellow", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("same_color", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("triangle", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("square", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("circle", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("same_shape", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_wall", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_corner", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_top", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_bottom", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_left", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_right", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("touch_object", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("above", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("below", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("top", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("bottom", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("small", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("medium", OBJECT_FILTER_TYPE)
name_mapper.map_name_with_signature("big", OBJECT_FILTER_TYPE)

name_mapper.map_name_with_signature("negate_filter", NEGATE_FILTER_TYPE)

# Adding numbers because they commonly occur in utterances. They're usually between 1 and 9. Since
# there are not too many of these productions, we're adding them to the global mapping instead of a
# local mapping in each world.
for num in range(1, 10):
    num_string = str(num)
    name_mapper.map_name_with_signature(name=num_string, signature=NUM_TYPE, alias=num_string)


COMMON_NAME_MAPPING = name_mapper.common_name_mapping
COMMON_TYPE_SIGNATURE = name_mapper.common_type_signature

from nltk.sem.logic import TRUTH_TYPE, EntityType, ComplexType, ANY_TYPE

from allennlp.data.semparse.type_declarations.type_declaration import NamedBasicType


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
NEGATE_FILTER_TYPE = ComplexType(ComplexType(OBJECT_TYPE, OBJECT_TYPE),
                                 ComplexType(OBJECT_TYPE, OBJECT_TYPE))
BOX_MEMBERSHIP_TYPE = ComplexType(BOX_TYPE, OBJECT_TYPE)
COLOR_FUNCTION_TYPE = ComplexType(OBJECT_TYPE, COLOR_TYPE)
SHAPE_FUNCTION_TYPE = ComplexType(OBJECT_TYPE, SHAPE_TYPE)

BOX_COLOR_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(ComplexType(BOX_TYPE, COLOR_TYPE),
                                                          ComplexType(COLOR_TYPE, BOX_TYPE)))
BOX_SHAPE_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(ComplexType(BOX_TYPE, SHAPE_TYPE),
                                                          ComplexType(SHAPE_TYPE, BOX_TYPE)))
BOX_COUNT_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(ComplexType(BOX_TYPE, NUM_TYPE),
                                                          ComplexType(NUM_TYPE, BOX_TYPE)))
ASSERT_COLOR_TYPE = ComplexType(OBJECT_TYPE, ComplexType(COLOR_TYPE, TRUTH_TYPE))
ASSERT_SHAPE_TYPE = ComplexType(OBJECT_TYPE, ComplexType(SHAPE_TYPE, TRUTH_TYPE))
ASSERT_BOX_COUNT_TYPE = ComplexType(BOX_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))
ASSERT_OBJECT_COUNT_TYPE = ComplexType(OBJECT_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))

BOX_EXISTS_TYPE = ComplexType(BOX_TYPE, TRUTH_TYPE)
OBJECT_EXISTS_TYPE = ComplexType(OBJECT_TYPE, TRUTH_TYPE)


COMMON_NAME_MAPPING = {"lambda": "\\", "x": "X"}
COMMON_TYPE_SIGNATURE = {"X": ANY_TYPE}

BASIC_TYPES = {NUM_TYPE, BOX_TYPE, OBJECT_TYPE, COLOR_TYPE, SHAPE_TYPE}


def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature


# Entities
add_common_name_with_type("all_objects", "O", OBJECT_TYPE)
add_common_name_with_type("all_boxes", "B", BOX_TYPE)
add_common_name_with_type("color_black", "C0", COLOR_TYPE)
add_common_name_with_type("color_blue", "C1", COLOR_TYPE)
add_common_name_with_type("color_yellow", "C2", COLOR_TYPE)
add_common_name_with_type("shape_triangle", "S0", SHAPE_TYPE)
add_common_name_with_type("shape_square", "S1", SHAPE_TYPE)
add_common_name_with_type("shape_circle", "S2", SHAPE_TYPE)


# Attribute functions
add_common_name_with_type("color", "C", COLOR_FUNCTION_TYPE)
add_common_name_with_type("shape", "S", SHAPE_FUNCTION_TYPE)
add_common_name_with_type("object_in_box", "I", BOX_MEMBERSHIP_TYPE)


# Assert functions
add_common_name_with_type("color_equals", "A0", ASSERT_COLOR_TYPE)
add_common_name_with_type("color_not_equals", "A1", ASSERT_COLOR_TYPE)
add_common_name_with_type("shape_equals", "A2", ASSERT_SHAPE_TYPE)
add_common_name_with_type("shape_not_equals", "A3", ASSERT_SHAPE_TYPE)
add_common_name_with_type("box_count_equals", "A4", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type("box_count_not_equals", "A5", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type("box_count_greater", "A6", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type("box_count_greater_equals", "A7", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type("box_count_lesser", "A8", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type("box_count_lesser_equals", "A9", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type("object_count_equals", "A10", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type("object_count_not_equals", "A11", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type("object_count_greater", "A12", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type("object_count_greater_equals", "A13", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type("object_count_lesser", "A14", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type("object_count_lesser_equals", "A15", ASSERT_OBJECT_COUNT_TYPE)


# Box filter functions
add_common_name_with_type("filter_count_equals", "F0", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type("filter_count_not_equals", "F1", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type("filter_shape_equals", "F2", BOX_SHAPE_FILTER_TYPE)
add_common_name_with_type("filter_shape_not_equals", "F3", BOX_SHAPE_FILTER_TYPE)
add_common_name_with_type("filter_color_equals", "F4", BOX_COLOR_FILTER_TYPE)
add_common_name_with_type("filter_color_not_equals", "F5", BOX_COLOR_FILTER_TYPE)
add_common_name_with_type("filter_count_greater", "F6", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type("filter_count_greater_equals", "F7", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type("filter_count_lesser", "F8", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type("filter_count_lesser_equals", "F9", BOX_COUNT_FILTER_TYPE)


# Object filter functions
add_common_name_with_type("black", "C3", OBJECT_FILTER_TYPE)
add_common_name_with_type("blue", "C4", OBJECT_FILTER_TYPE)
add_common_name_with_type("yellow", "C5", OBJECT_FILTER_TYPE)
add_common_name_with_type("same_color", "C6", OBJECT_FILTER_TYPE)
add_common_name_with_type("triangle", "S3", OBJECT_FILTER_TYPE)
add_common_name_with_type("square", "S4", OBJECT_FILTER_TYPE)
add_common_name_with_type("circle", "S5", OBJECT_FILTER_TYPE)
add_common_name_with_type("same_shape", "S6", OBJECT_FILTER_TYPE)
add_common_name_with_type("touch_wall", "T0", OBJECT_FILTER_TYPE)
add_common_name_with_type("touch_corner", "T1", OBJECT_FILTER_TYPE)
add_common_name_with_type("touch_top", "T2", OBJECT_FILTER_TYPE)
add_common_name_with_type("touch_bottom", "T3", OBJECT_FILTER_TYPE)
add_common_name_with_type("touch_left", "T4", OBJECT_FILTER_TYPE)
add_common_name_with_type("touch_right", "T5", OBJECT_FILTER_TYPE)
add_common_name_with_type("above", "L0", OBJECT_FILTER_TYPE)
add_common_name_with_type("below", "L1", OBJECT_FILTER_TYPE)
add_common_name_with_type("top", "L2", OBJECT_FILTER_TYPE)
add_common_name_with_type("bottom", "L3", OBJECT_FILTER_TYPE)
add_common_name_with_type("small", "Z0", OBJECT_FILTER_TYPE)
add_common_name_with_type("medium", "Z1", OBJECT_FILTER_TYPE)
add_common_name_with_type("big", "Z2", OBJECT_FILTER_TYPE)

add_common_name_with_type("box_exists", "E0", BOX_EXISTS_TYPE)
add_common_name_with_type("object_exists", "E1", OBJECT_EXISTS_TYPE)

add_common_name_with_type("negate_filter", "N", NEGATE_FILTER_TYPE)

# Adding numbers because they commonly occur in utterances. They're usually between 1 and 9. Since
# there are not too many of these productions, we're adding them to the global mapping instead of a
# local mapping in each world.
for num in range(1, 10):
    num_string = str(num)
    add_common_name_with_type(num_string, num_string, NUM_TYPE)

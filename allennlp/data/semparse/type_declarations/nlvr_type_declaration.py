from typing import Optional
from overrides import overrides

from nltk.sem.logic import TRUTH_TYPE, EntityType, Type, ComplexType, ANY_TYPE

from allennlp.data.semparse.type_declarations.type_declaration import NamedBasicType, PlaceholderType


class BoxFilterType(PlaceholderType):
    """
    This is the type of the functions that filter boxes. The corresponding Python function takes three
    arguments, and looks like ``filter(initial_set_of_boxes, attribute_function, target_attribute)`` .
    Hence, the signature of the filter function is <b,<<b,#1>,<#1,b>>>.
    """
    @property
    def _signature(self) -> str:
        return "<b,<<b,#1>,<#1,b>>>"

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        if not isinstance(other, ComplexType):
            return None
        if not isinstance(other.second, ComplexType):
            return None
        if not isinstance(other.second.first, ComplexType) or not isinstance(other.second.second, ComplexType):
            return None
        resolved_type = other.resolve(ComplexType(BOX_TYPE, ComplexType(ComplexType(BOX_TYPE, ANY_TYPE),
                                                                        ComplexType(ANY_TYPE, BOX_TYPE))))
        if not resolved_type:
            return None
        first_placeholder = resolved_type.second.first.second
        second_placeholder = resolved_type.second.second.first
        resolved_placeholder = first_placeholder.resolve(second_placeholder)
        if not resolved_placeholder:
            return None
        return BoxFilterType(BOX_TYPE, ComplexType(ComplexType(BOX_TYPE, resolved_placeholder),
                                                   ComplexType(resolved_placeholder, BOX_TYPE)))

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return self.second


class AssertType(PlaceholderType):
    """
    This is the type of assert operations. Corresponding Python function looks like
    ``assert(attribute, target_attribute)`` , where the attribute and target attribute can either be numbers,
    shapes or colors, and the return type is boolean. So the signature of the function is <#1,<#1,t>>.
    """
    @property
    def _signature(self) -> str:
        return "<#1,<#1,t>>"

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        if not isinstance(other, ComplexType):
            return None
        if not isinstance(other.second, ComplexType):
            return None
        resolved_type = other.resolve(ComplexType(ANY_TYPE, ComplexType(ANY_TYPE, TRUTH_TYPE)))
        if not resolved_type:
            return None
        resolved_placeholder = resolved_type.first.resolve(resolved_type.second.first)
        if not resolved_placeholder:
            return None
        return AssertType(resolved_placeholder, ComplexType(resolved_type, TRUTH_TYPE))

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return ComplexType(argument_type, TRUTH_TYPE)


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
BOX_FILTER_TYPE = BoxFilterType(BOX_TYPE, ComplexType(ComplexType(BOX_TYPE, ANY_TYPE),
                                                      ComplexType(ANY_TYPE, BOX_TYPE)))
ASSERT_TYPE = AssertType(ANY_TYPE, ComplexType(ANY_TYPE, TRUTH_TYPE))


COMMON_NAME_MAPPING = {}
COMMON_TYPE_SIGNATURE = {}


def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature


add_common_name_with_type("all_objects", "O", OBJECT_TYPE)
add_common_name_with_type("all_boxes", "B", BOX_TYPE)


# Attribute functions
add_common_name_with_type("color", "C", COLOR_FUNCTION_TYPE)
add_common_name_with_type("shape", "S", SHAPE_FUNCTION_TYPE)
add_common_name_with_type("object_in_box", "I", BOX_MEMBERSHIP_TYPE)


# Assert functions
add_common_name_with_type("assert_equals", "A0", ASSERT_TYPE)
add_common_name_with_type("assert_not_equals", "A1", ASSERT_TYPE)
add_common_name_with_type("assert_greater", "A2", ASSERT_TYPE)
add_common_name_with_type("assert_greater_equals", "A3", ASSERT_TYPE)
add_common_name_with_type("assert_lesser", "A4", ASSERT_TYPE)
add_common_name_with_type("assert_lesser_equals", "A5", ASSERT_TYPE)


# Box filter functions
add_common_name_with_type("filter_equals", "F0", BOX_FILTER_TYPE)
add_common_name_with_type("filter_not_equals", "F1", BOX_FILTER_TYPE)
add_common_name_with_type("filter_greater", "F2", BOX_FILTER_TYPE)
add_common_name_with_type("filter_greater_equals", "F3", BOX_FILTER_TYPE)
add_common_name_with_type("filter_lesser", "F4", BOX_FILTER_TYPE)
add_common_name_with_type("filter_lesser_equals", "F5", BOX_FILTER_TYPE)


# Object filter functions
add_common_name_with_type("black", "C0", OBJECT_FILTER_TYPE)
add_common_name_with_type("blue", "C1", OBJECT_FILTER_TYPE)
add_common_name_with_type("yellow", "C2", OBJECT_FILTER_TYPE)
add_common_name_with_type("same_color", "C3", OBJECT_FILTER_TYPE)
add_common_name_with_type("triangle", "S0", OBJECT_FILTER_TYPE)
add_common_name_with_type("square", "S1", OBJECT_FILTER_TYPE)
add_common_name_with_type("circle", "S2", OBJECT_FILTER_TYPE)
add_common_name_with_type("same_shape", "S3", OBJECT_FILTER_TYPE)
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

add_common_name_with_type("negate_filter", "N", NEGATE_FILTER_TYPE)

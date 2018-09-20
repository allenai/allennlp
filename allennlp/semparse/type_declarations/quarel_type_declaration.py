"""
Defines all the types in the QuaRel domain.
"""
from typing import Dict
from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType, Type

class QuarelTypeDeclaration:
    def __init__(self, syntax: str) -> None:

        self.COMMON_NAME_MAPPING: Dict[str, str] = {}

        self.COMMON_TYPE_SIGNATURE: Dict[str, Type] = {}

        NUM_TYPE = NamedBasicType("NUM")
        ATTR_TYPE = NamedBasicType("ATTR")
        RDIR_TYPE = NamedBasicType("RDIR")
        WORLD_TYPE = NamedBasicType("WORLD")
        VAR_TYPE = NamedBasicType("VAR")

        self.BASIC_TYPES = {NUM_TYPE, ATTR_TYPE, RDIR_TYPE, WORLD_TYPE, VAR_TYPE}

        if syntax == "quarel_friction":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                              ComplexType(ATTR_TYPE,
                                                          ComplexType(ATTR_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            # Attributes
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", RDIR_TYPE)
            self.add_common_name_with_type("low", "R13", RDIR_TYPE)
            self.add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                    ATTR_FUNCTION_TYPE: 2,
                    INFER_FUNCTION_TYPE: 3,
                    AND_FUNCTION_TYPE: 2
            }
        elif syntax == "quarel_v1_attr_entities" or syntax == "quarel_friction_attr_entities":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                              ComplexType(ATTR_TYPE,
                                                          ComplexType(ATTR_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            # TODO: Remove this?
            self.add_common_name_with_type("placeholder", "A99", ATTR_FUNCTION_TYPE)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", RDIR_TYPE)
            self.add_common_name_with_type("low", "R13", RDIR_TYPE)
            self.add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                    ATTR_FUNCTION_TYPE: 2,
                    INFER_FUNCTION_TYPE: 3,
                    AND_FUNCTION_TYPE: 2
            }

        elif syntax == "quarel_v1":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                              ComplexType(ATTR_TYPE,
                                                          ComplexType(ATTR_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            # Attributes
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("acceleration", "A15", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("amountSweat", "A16", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("apparentSize", "A17", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("breakability", "A18", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("brightness", "A19", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("exerciseIntensity", "A20", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("flexibility", "A21", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("gravity", "A22", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("loudness", "A23", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("mass", "A24", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("strength", "A25", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("thickness", "A26", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("time", "A27", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("weight", "A28", ATTR_FUNCTION_TYPE)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", RDIR_TYPE)
            self.add_common_name_with_type("low", "R13", RDIR_TYPE)
            self.add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                    ATTR_FUNCTION_TYPE: 2,
                    INFER_FUNCTION_TYPE: 3,
                    AND_FUNCTION_TYPE: 2
            }

        else:
            raise Exception(f"Unknown LF syntax specification: {syntax}")

        self.add_common_name_with_type("higher", "R10", RDIR_TYPE)
        self.add_common_name_with_type("lower", "R11", RDIR_TYPE)

        self.add_common_name_with_type("world1", "W11", WORLD_TYPE)
        self.add_common_name_with_type("world2", "W12", WORLD_TYPE)

        # Hack to expose types
        self.WORLD_TYPE = WORLD_TYPE
        self.ATTR_FUNCTION_TYPE = ATTR_FUNCTION_TYPE
        self.VAR_TYPE = VAR_TYPE


        self.STARTING_TYPES = {NUM_TYPE}

    def add_common_name_with_type(self, name: str, mapping: str, type_signature: Type) -> None:
        self.COMMON_NAME_MAPPING[name] = mapping
        self.COMMON_TYPE_SIGNATURE[mapping] = type_signature

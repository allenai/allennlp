"""
Defines all the types in the QuaRel domain.
"""
from typing import Dict
from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType, Type

class QuarelTypeDeclaration:
    def __init__(self, syntax: str) -> None:

        self.common_name_mapping: Dict[str, str] = {}

        self.common_type_signature: Dict[str, Type] = {}

        num_type = NamedBasicType("NUM")
        attr_type = NamedBasicType("ATTR")
        rdir_type = NamedBasicType("RDIR")
        world_type = NamedBasicType("WORLD")
        var_type = NamedBasicType("VAR")

        self.basic_types = {num_type, attr_type, rdir_type, world_type, var_type}

        if syntax == "quarel_friction":
            # attributes: <<QDIR, <WORLD, ATTR>>
            attr_function_type = ComplexType(rdir_type,
                                             ComplexType(world_type, attr_type))

            and_function_type = ComplexType(attr_type, ComplexType(attr_type, attr_type))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            infer_function_type = ComplexType(attr_type,
                                              ComplexType(attr_type,
                                                          ComplexType(attr_type, num_type)))
            self.add_common_name_with_type("infer", "I10", infer_function_type)
            # Attributes
            self.add_common_name_with_type("friction", "A10", attr_function_type)
            self.add_common_name_with_type("smoothness", "A11", attr_function_type)
            self.add_common_name_with_type("speed", "A12", attr_function_type)
            self.add_common_name_with_type("heat", "A13", attr_function_type)
            self.add_common_name_with_type("distance", "A14", attr_function_type)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", rdir_type)
            self.add_common_name_with_type("low", "R13", rdir_type)
            self.add_common_name_with_type("and", "C10", and_function_type)

            self.curried_functions = {
                    attr_function_type: 2,
                    infer_function_type: 3,
                    and_function_type: 2
            }
        elif syntax == "quarel_v1_attr_entities" or syntax == "quarel_friction_attr_entities":
            # attributes: <<QDIR, <WORLD, ATTR>>
            attr_function_type = ComplexType(rdir_type,
                                             ComplexType(world_type, attr_type))

            and_function_type = ComplexType(attr_type, ComplexType(attr_type, attr_type))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            infer_function_type = ComplexType(attr_type,
                                              ComplexType(attr_type,
                                                          ComplexType(attr_type, num_type)))
            self.add_common_name_with_type("infer", "I10", infer_function_type)
            # TODO: Remove this?
            self.add_common_name_with_type("placeholder", "A99", attr_function_type)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", rdir_type)
            self.add_common_name_with_type("low", "R13", rdir_type)
            self.add_common_name_with_type("and", "C10", and_function_type)

            self.curried_functions = {
                    attr_function_type: 2,
                    infer_function_type: 3,
                    and_function_type: 2
            }

        elif syntax == "quarel_v1":
            # attributes: <<QDIR, <WORLD, ATTR>>
            attr_function_type = ComplexType(rdir_type,
                                             ComplexType(world_type, attr_type))

            and_function_type = ComplexType(attr_type, ComplexType(attr_type, attr_type))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            infer_function_type = ComplexType(attr_type,
                                              ComplexType(attr_type,
                                                          ComplexType(attr_type, num_type)))
            self.add_common_name_with_type("infer", "I10", infer_function_type)
            # Attributes
            self.add_common_name_with_type("friction", "A10", attr_function_type)
            self.add_common_name_with_type("smoothness", "A11", attr_function_type)
            self.add_common_name_with_type("speed", "A12", attr_function_type)
            self.add_common_name_with_type("heat", "A13", attr_function_type)
            self.add_common_name_with_type("distance", "A14", attr_function_type)
            self.add_common_name_with_type("acceleration", "A15", attr_function_type)
            self.add_common_name_with_type("amountSweat", "A16", attr_function_type)
            self.add_common_name_with_type("apparentSize", "A17", attr_function_type)
            self.add_common_name_with_type("breakability", "A18", attr_function_type)
            self.add_common_name_with_type("brightness", "A19", attr_function_type)
            self.add_common_name_with_type("exerciseIntensity", "A20", attr_function_type)
            self.add_common_name_with_type("flexibility", "A21", attr_function_type)
            self.add_common_name_with_type("gravity", "A22", attr_function_type)
            self.add_common_name_with_type("loudness", "A23", attr_function_type)
            self.add_common_name_with_type("mass", "A24", attr_function_type)
            self.add_common_name_with_type("strength", "A25", attr_function_type)
            self.add_common_name_with_type("thickness", "A26", attr_function_type)
            self.add_common_name_with_type("time", "A27", attr_function_type)
            self.add_common_name_with_type("weight", "A28", attr_function_type)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", rdir_type)
            self.add_common_name_with_type("low", "R13", rdir_type)
            self.add_common_name_with_type("and", "C10", and_function_type)

            self.curried_functions = {
                    attr_function_type: 2,
                    infer_function_type: 3,
                    and_function_type: 2
            }

        else:
            raise Exception(f"Unknown LF syntax specification: {syntax}")

        self.add_common_name_with_type("higher", "R10", rdir_type)
        self.add_common_name_with_type("lower", "R11", rdir_type)

        self.add_common_name_with_type("world1", "W11", world_type)
        self.add_common_name_with_type("world2", "W12", world_type)

        # Hack to expose types
        self.world_type = world_type
        self.attr_function_type = attr_function_type
        self.var_type = var_type

        self.starting_types = {num_type}

    def add_common_name_with_type(self, name: str, mapping: str, type_signature: Type) -> None:
        self.common_name_mapping[name] = mapping
        self.common_type_signature[mapping] = type_signature

"""
Defines all the types in the QuaRel domain.
"""
from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType, NameMapper

class QuarelTypeDeclaration:
    def __init__(self, syntax: str) -> None:

        self.name_mapper = NameMapper()

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
            self.name_mapper.map_name_with_signature("infer", infer_function_type)
            # Attributes
            self.name_mapper.map_name_with_signature("friction", attr_function_type)
            self.name_mapper.map_name_with_signature("smoothness", attr_function_type)
            self.name_mapper.map_name_with_signature("speed", attr_function_type)
            self.name_mapper.map_name_with_signature("heat", attr_function_type)
            self.name_mapper.map_name_with_signature("distance", attr_function_type)

            # For simplicity we treat "high" and "low" as directions as well
            self.name_mapper.map_name_with_signature("high", rdir_type)
            self.name_mapper.map_name_with_signature("low", rdir_type)
            self.name_mapper.map_name_with_signature("and", and_function_type)

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
            self.name_mapper.map_name_with_signature("infer", infer_function_type)
            # TODO: Remove this?
            self.name_mapper.map_name_with_signature("placeholder", attr_function_type)

            # For simplicity we treat "high" and "low" as directions as well
            self.name_mapper.map_name_with_signature("high", rdir_type)
            self.name_mapper.map_name_with_signature("low", rdir_type)
            self.name_mapper.map_name_with_signature("and", and_function_type)

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
            self.name_mapper.map_name_with_signature("infer", infer_function_type)
            # Attributes
            self.name_mapper.map_name_with_signature("friction", attr_function_type)
            self.name_mapper.map_name_with_signature("smoothness", attr_function_type)
            self.name_mapper.map_name_with_signature("speed", attr_function_type)
            self.name_mapper.map_name_with_signature("heat", attr_function_type)
            self.name_mapper.map_name_with_signature("distance", attr_function_type)
            self.name_mapper.map_name_with_signature("acceleration", attr_function_type)
            self.name_mapper.map_name_with_signature("amountSweat", attr_function_type)
            self.name_mapper.map_name_with_signature("apparentSize", attr_function_type)
            self.name_mapper.map_name_with_signature("breakability", attr_function_type)
            self.name_mapper.map_name_with_signature("brightness", attr_function_type)
            self.name_mapper.map_name_with_signature("exerciseIntensity", attr_function_type)
            self.name_mapper.map_name_with_signature("flexibility", attr_function_type)
            self.name_mapper.map_name_with_signature("gravity", attr_function_type)
            self.name_mapper.map_name_with_signature("loudness", attr_function_type)
            self.name_mapper.map_name_with_signature("mass", attr_function_type)
            self.name_mapper.map_name_with_signature("strength", attr_function_type)
            self.name_mapper.map_name_with_signature("thickness", attr_function_type)
            self.name_mapper.map_name_with_signature("time", attr_function_type)
            self.name_mapper.map_name_with_signature("weight", attr_function_type)

            # For simplicity we treat "high" and "low" as directions as well
            self.name_mapper.map_name_with_signature("high", rdir_type)
            self.name_mapper.map_name_with_signature("low", rdir_type)
            self.name_mapper.map_name_with_signature("and", and_function_type)

            self.curried_functions = {
                    attr_function_type: 2,
                    infer_function_type: 3,
                    and_function_type: 2
            }

        else:
            raise Exception(f"Unknown LF syntax specification: {syntax}")

        self.name_mapper.map_name_with_signature("higher", rdir_type)
        self.name_mapper.map_name_with_signature("lower", rdir_type)

        self.name_mapper.map_name_with_signature("world1", world_type)
        self.name_mapper.map_name_with_signature("world2", world_type)

        # Hack to expose types
        self.world_type = world_type
        self.attr_function_type = attr_function_type
        self.var_type = var_type

        self.starting_types = {num_type}

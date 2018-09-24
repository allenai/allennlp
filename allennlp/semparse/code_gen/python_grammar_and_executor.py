"""
1. Creating the Parsimonious grammar.
    a. Extract type annotations, cluster and implement type hierarchy.
    b. Create the grammar in a programmatic, reversable way such that
       the executor comes for free.

Notes
- args and kwargs are not allowed.
- mypy must pass first?
- no default arguments
- no generator function

2. Wrap provided functions and classes in an Executor which takes the
   generated grammar and automatically executes things.

"""
from typing import Tuple, List, Set
import importlib
import importlib.util
import inspect
from parsimonious import Grammar

from allennlp.common.registrable import Registrable

class Executor:

    def __init__(self, grammar: Grammar):

        self.grammar = grammar


TypeSignature = Tuple[Tuple[type], type]

class GrammarFunction:
    def __init__(self, name: str, signature: TypeSignature, func):
        self.name = name
        self.signature = signature
        self.func = func

    def get_all_types(self) -> Set[type]:
        return_signatures = set(self.signature[0])
        return_signatures.add(self.signature[1])
        return return_signatures

class GrammarClass:
    def __init__(self, name: str, signature: TypeSignature, methods: List[GrammarFunction]):
        self.name = name
        self.signature = signature
        self.methods = methods

    def get_all_types(self):
        return_signatures = set(self.signature[0])
        return_signatures.add(self.signature[1])

        for func in self.methods:
            return_signatures.add(func.signature[1])
            return_signatures.update(func.signature[0])
        return return_signatures 


class Inspector:

    def __init__(self, file_path: str) -> None:

        spec = importlib.util.spec_from_file_location("grammar", file_path)
        grammar_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grammar_module)
        self._grammar_module = grammar_module

        # TODO: Don't hard code the fact that it's called top_level_grammar
        self._top_level_grammar = self._grammar_module.top_level_grammar

    def get_type_signature_from_func(self, signature: inspect.Signature) -> TypeSignature:

        arguments = [param.annotation for param in signature.parameters.values()]
        return_type = signature.return_annotation
        return (tuple(arguments), return_type)

    def get_functions_and_classes(self) -> Tuple[List[GrammarFunction], List[GrammarClass]]:
        functions: List[GrammarFunction] = []
        classes: List[GrammarClass] = []
        # Loop over all objects in the API of the grammar.
        for name in self._grammar_module.__all__:
            obj = getattr(self._grammar_module, name)

            # currently we have two cases - either the object is
            # a function or a class. In the first case, we want it's signature;
            # in the second, we want the signature of it's constructor and all
            # it's public methods.
            if inspect.isfunction(obj):
                print("Function: ", name)
                signature = inspect.signature(obj)

                type_signature = self.get_type_signature_from_func(signature)
                functions.append(GrammarFunction(name, type_signature, obj))

            elif inspect.isclass(obj):
                print("Class: ", name)
                # Get all the methods on the class.
                members = inspect.getmembers(obj, predicate=inspect.isfunction)
                # Filter to exclude magic method rubbish we don't need.
                members = [x for x in members if "__" not in x[0]]

                methods: List[GrammarFunction] = []
                for method_name, method in members:
                    method_signature = self.get_type_signature_from_func(inspect.signature(method))
                    methods.append(GrammarFunction(method_name, method_signature, method))

                constructor_signature = self.get_type_signature_from_func(inspect.signature(obj.__init__))
                classes.append(GrammarClass(name, constructor_signature, methods))

        return functions, classes

    
    def create_grammar_from_spec(self,
                                 functions: List[GrammarFunction],
                                 classes: List[GrammarClass]) -> Grammar:
        
        all_base_types = set()
        for x in functions + classes:
            all_base_types.update(x.get_all_types())

        print(all_base_types)




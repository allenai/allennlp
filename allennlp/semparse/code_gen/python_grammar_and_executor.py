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

import ast
import importlib
import importlib.util
import inspect
from parsimonious import Grammar

class Executor:

    def __init__(self, grammar: Grammar):

        self.grammar = grammar


class Inspector:

    def __init__(self, file_path: str) -> None:

        spec = importlib.util.spec_from_file_location("grammar", file_path)
        grammar_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grammar_module)

        print(grammar_module.expand)

        for name in grammar_module.__all__:
            print(getattr(grammar_module, name))

        self._grammar_module = grammar_module

    def get_functions_and_classes(self):
        # Loop over all objects in the API of the grammar.
        for name in self._grammar_module.__all__:
            print(name)
            obj = getattr(self._grammar_module, name)
            if inspect.isfunction(obj):
                print(inspect.signature(obj))

            elif inspect.isclass(obj):
                # Get all the methods on the class.
                members = inspect.getmembers(obj, predicate=lambda x: not inspect.ismethod(x))
                # Filter to exclude magic method rubbish we don't need.
                members = [x for x in members if "__" not in x[0]]
                print("All members: ", members)
                for member in members:
                    print(member[0])
                    print(inspect.signature(member[1]))


                print(inspect.signature(obj.__init__))

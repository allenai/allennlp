

from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.code_gen.python_grammar_and_executor import Inspector


class TestInspector(AllenNlpTestCase):

    def test_inspector_can_extract_functions(self):

        inspector = Inspector("allennlp/tests/semparse/code_gen/baby_grammar.py")
        functions, classes = inspector.get_functions_and_classes()
        inspector.create_grammar_from_spec(functions, classes)
        print(functions)
        print(classes)

# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.semparse.knowledge_graphs import QuestionKnowledgeGraph


class TestQuestionKnowledgeGraph(AllenNlpTestCase):
    def test_numeric_entities(self):
        question_tokens = [Token(x) for x in
                           ['I', 'have', 'a', '20', 'dollar', 'bill', 'and', 'a', '50', 'dollar', 'bill']]
        question_knowledge_graph = QuestionKnowledgeGraph.read(question_tokens)
        assert(question_knowledge_graph.entities == ['20', '50'])


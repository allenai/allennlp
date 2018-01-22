# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse import ParsingError
from allennlp.data.semparse.knowledge_graphs.table_knowledge_graph import TableKnowledgeGraph
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.data.tokenizers import Token


class TestWorld(AllenNlpTestCase):
    # Most of these tests will use a WikiTablesWorld as a concrete instance to test the base class
    # methods.
    def setUp(self):
        super().setUp()
        self.table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/sample_table.tsv")
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '?']]
        self.world = WikiTablesWorld(self.table_kg, question_tokens)

    def test_lisp_to_nested_expression(self):
        logical_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world._lisp_to_nested_expression(logical_form)
        assert expression == [[['reverse', 'fb:row.row.year'], ['fb:row.row.league', 'fb:cell.usl_a_league']]]
        logical_form = "(count (and (division 1) (tier (!= null))))"
        expression = self.world._lisp_to_nested_expression(logical_form)
        assert expression == [['count', ['and', ['division', '1'], ['tier', ['!=', 'null']]]]]

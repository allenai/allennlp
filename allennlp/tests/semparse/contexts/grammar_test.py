

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.contexts.grammars import SQL_GRAMMAR2

class GrammarTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()

        self.grammar = SQL_GRAMMAR2


    def test_grammar_can_parse_order_by(self):
        sql_string = "SELECT CITYalias0.STATE_NAME FROM CITY AS CITYalias0 GROUP BY CITYalias0.STATE_NAME ORDER BY SUM (CITYalias0.POPULATION) DESC LIMIT 1 ;"

        self.grammar.parse(sql_string)

    def test_grammar_can_parse_in_statements(self):
        sql_string = "SELECT STATEalias0.CAPITAL FROM STATE AS STATEalias0 WHERE STATEalias0.STATE_NAME IN (SELECT BORDER_INFOalias0.BORDER FROM BORDER_INFO AS BORDER_INFOalias0 GROUP BY BORDER_INFOalias0.BORDER HAVING COUNT (1) = (SELECT MAX (DERIVED_TABLEalias0.DERIVED_FIELDalias0) FROM (SELECT BORDER_INFOalias1.BORDER, COUNT (1) AS DERIVED_FIELDalias0 FROM BORDER_INFO AS BORDER_INFOalias1 GROUP BY BORDER_INFOalias1.BORDER) DERIVED_TABLEalias0)) ;"

        self.grammar.parse(sql_string)
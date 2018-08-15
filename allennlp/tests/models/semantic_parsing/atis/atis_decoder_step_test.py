# pylint: disable=invalid-name,no-self-use,protected-access
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import SimilarityFunction
from allennlp.nn.decoding import AtisGrammarState, GrammarState, RnnState

class AtisDecoderStepTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

    def test_atis_grammar_state(self):
        valid_actions = None

        action_seq = ['stmt -> [query, ";"]', 'query -> [lparen, "SELECT", distinct, select_results, "FROM", table_refs, where_clause, rparen]', 'rparen -> [")"]', 'where_clause -> ["WHERE", lparen, conditions, rparen]', 'rparen -> [")"]', 'conditions -> [condition]', 'condition -> [biexpr]', 'biexpr -> [col_ref, binaryop, value]', 'value -> [pos_value]', 'pos_value -> [string]', 'string -> ["\'BOSTON\'"]', 'binaryop -> ["="]', 'col_ref -> ["city", ".", "city_name"]', 'lparen -> ["("]', 'table_refs -> [table_name]', 'table_name -> ["city"]', 'select_results -> [col_refs]', 'col_refs -> [col_ref, ",", col_refs]', 'col_refs -> [col_ref]', 'col_ref -> ["city", ".", "city_name"]', 'col_ref -> ["city", ".", "city_code"]', 'distinct -> ["DISTINCT"]', 'lparen -> ["("]']        

        grammar_state = AtisGrammarState(['stmt'], {}, valid_actions, {}, is_nonterminal)
        for action in action_seq:
            grammar_state = grammar_state.take_action(action)
            print('Stack', grammar_state._nonterminal_stack)
        assert grammar_state._nonterminal_stack == []




        
        



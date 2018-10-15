# pylint: disable=invalid-name,no-self-use,protected-access
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import SimilarityFunction
from allennlp.state_machines.states import GrammarStatelet
from allennlp.models.semantic_parsing.atis.atis_semantic_parser import AtisSemanticParser
from allennlp.semparse.worlds import AtisWorld

class AtisGrammarStateletTest(AllenNlpTestCase):
    def test_atis_grammar_statelet(self):
        valid_actions = None
        world = AtisWorld([("give me all flights from boston to "
                            "philadelphia next week arriving after lunch")])
        action_sequence = \
                ['statement -> [query, ";"]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'distinct -> ["DISTINCT"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref, ",", col_refs]',
                 'col_ref -> ["city", ".", "city_code"]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["city", ".", "city_name"]',
                 'table_refs -> [table_name]',
                 'table_name -> ["city"]',
                 'where_clause -> ["WHERE", "(", conditions, ")"]',
                 'conditions -> [condition]',
                 'condition -> [biexpr]',
                 'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                 'binaryop -> ["="]',
                 'city_city_name_string -> ["\'BOSTON\'"]']

        grammar_state = GrammarStatelet(['statement'],
                                        world.valid_actions,
                                        AtisSemanticParser.is_nonterminal)
        for action in action_sequence:
            grammar_state = grammar_state.take_action(action)
        assert grammar_state._nonterminal_stack == []





# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy

from allennlp.data import Vocabulary
from allennlp.data.fields import ProductionRuleField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer

from allennlp.common.testing import AllenNlpTestCase


class TestProductionRuleField(AllenNlpTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.s_rule_index = self.vocab.add_token_to_namespace("S", namespace='rule_labels')
        self.np_index = self.vocab.add_token_to_namespace("NP", namespace='rule_labels')
        self.vp_index = self.vocab.add_token_to_namespace("VP", namespace='rule_labels')
        self.np_vp_index = self.vocab.add_token_to_namespace("[NP, VP]", namespace='rule_labels')
        self.identity_index = self.vocab.add_token_to_namespace("identity", namespace='entities')
        self.t_index = self.vocab.add_token_to_namespace("t", namespace='characters')
        self.e_index = self.vocab.add_token_to_namespace("e", namespace='characters')
        self.s_index = self.vocab.add_token_to_namespace("s", namespace='characters')

        self.oov_index = self.vocab.get_token_index('random OOV string', namespace='entities')

        self.terminal_indexers = {"entities": SingleIdTokenIndexer("entities")}
        self.nonterminal_indexers = {"rules": SingleIdTokenIndexer("rule_labels")}
        self.character_indexer = {'characters': TokenCharactersIndexer('characters')}
        self.nonterminal_types = {'S', 'NP', 'VP'}

        super(TestProductionRuleField, self).setUp()

    def _make_field(self,
                    rule_string: str,
                    terminal_indexers=None,
                    nonterminal_indexers=None) -> ProductionRuleField:
        terminal_indexers = terminal_indexers or self.terminal_indexers
        nonterminal_indexers = nonterminal_indexers or self.nonterminal_indexers
        return ProductionRuleField(rule_string,
                                   terminal_indexers=terminal_indexers,
                                   nonterminal_indexers=nonterminal_indexers,
                                   nonterminal_types=self.nonterminal_types)

    def test_field_counts_vocab_items_correctly(self):
        field = self._make_field('S -> [NP, VP]')
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["rule_labels"]["S"] == 1
        assert namespace_token_counts["rule_labels"]["[NP, VP]"] == 1
        assert len(namespace_token_counts["rule_labels"]) == 2
        assert list(namespace_token_counts.keys()) == ["rule_labels"]

        field = self._make_field('<e,e> -> identity')
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["rule_labels"]["<e,e>"] == 1
        assert len(namespace_token_counts["rule_labels"]) == 1
        assert namespace_token_counts["entities"]["identity"] == 1
        assert len(namespace_token_counts["entities"]) == 1
        assert set(namespace_token_counts.keys()) == {"rule_labels", "entities"}

        field = self._make_field('S -> VP')
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["rule_labels"]["S"] == 1
        assert namespace_token_counts["rule_labels"]["VP"] == 1
        assert len(namespace_token_counts["rule_labels"]) == 2
        assert set(namespace_token_counts.keys()) == {"rule_labels"}

    def test_index_converts_field_correctly(self):
        # pylint: disable=protected-access
        field = self._make_field('S -> [NP, VP]')
        field.index(self.vocab)
        assert field._indexed_left_side == {'rules': self.s_rule_index}
        assert field._indexed_right_side == {'rules': self.np_vp_index}

        field = self._make_field('VP -> eats')
        field.index(self.vocab)
        assert field._indexed_left_side == {'rules': self.vp_index}
        assert field._indexed_right_side == {'entities': self.oov_index}

        field = self._make_field('NP -> VP')
        field.index(self.vocab)
        assert field._indexed_left_side == {'rules': self.np_index}
        assert field._indexed_right_side == {'rules': self.vp_index}

        field = self._make_field('S -> identity')
        field.index(self.vocab)
        assert field._indexed_left_side == {'rules': self.s_rule_index}
        assert field._indexed_right_side == {'entities': self.identity_index}

        # pylint: enable=protected-access

    def test_get_padding_lengths_raises_if_not_indexed(self):
        field = self._make_field('S -> [NP, VP]')
        with pytest.raises(RuntimeError):
            field.get_padding_lengths()

    def test_padding_lengths_are_computed_correctly(self):
        field = self._make_field('S -> [NP, VP]')
        field.index(self.vocab)
        assert field.get_padding_lengths() == {}

        field = self._make_field('S -> test', terminal_indexers=self.character_indexer)
        field.index(self.vocab)
        assert field.get_padding_lengths() == {'num_token_characters': 4}

        field = self._make_field('S -> test', nonterminal_indexers=self.character_indexer)
        field.index(self.vocab)
        assert field.get_padding_lengths() == {'num_token_characters': 1}

        field = self._make_field('S -> test',
                                 terminal_indexers=self.character_indexer,
                                 nonterminal_indexers=self.character_indexer)
        field.index(self.vocab)
        assert field.get_padding_lengths() == {'num_token_characters': 4}

    def test_as_array_produces_correct_output(self):
        field = self._make_field('S -> [NP, VP]')
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        array_dict = field.as_array(padding_lengths)
        assert array_dict.keys() == {'left', 'right'}
        assert array_dict['left'] == ('S', True, {'rules': numpy.array(self.s_rule_index)})
        assert array_dict['right'] == ('[NP, VP]', True, {'rules': numpy.array(self.np_vp_index)})

        field = self._make_field('NP -> test', terminal_indexers=self.character_indexer)
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        array_dict = field.as_array(padding_lengths)
        assert array_dict.keys() == {'left', 'right'}
        assert array_dict['left'] == ('NP', True, {'rules': numpy.array(self.np_index)})
        assert array_dict['right'][0] == 'test'
        assert array_dict['right'][1] is False
        numpy.testing.assert_array_equal(array_dict['right'][2]['characters'],
                                         numpy.array([self.t_index, self.e_index,
                                                      self.s_index, self.t_index]))

    def test_batch_arrays_does_not_modify_list(self):
        field = self._make_field('S -> [NP, VP]')
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        array_dict1 = field.as_array(padding_lengths)

        field = self._make_field('S -> test')
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        array_dict2 = field.as_array(padding_lengths)
        array_list = [array_dict1, array_dict2]
        assert field.batch_arrays(array_list) == array_list

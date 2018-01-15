# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy
from numpy.testing import assert_array_equal

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.fields import ListField, ProductionRuleField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer


class TestProductionRuleField(AllenNlpTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.s_rule_index = self.vocab.add_token_to_namespace("S", namespace='rule_labels')
        self.np_index = self.vocab.add_token_to_namespace("NP", namespace='rule_labels')
        self.vp_index = self.vocab.add_token_to_namespace("VP", namespace='rule_labels')
        self.np_vp_index = self.vocab.add_token_to_namespace("[NP, VP]", namespace='rule_labels')
        self.identity_index = self.vocab.add_token_to_namespace("identity", namespace='entities')
        self.a_index = self.vocab.add_token_to_namespace("a", namespace='characters')
        self.c_index = self.vocab.add_token_to_namespace("c", namespace='characters')
        self.e_index = self.vocab.add_token_to_namespace("e", namespace='characters')
        self.s_index = self.vocab.add_token_to_namespace("s", namespace='characters')
        self.t_index = self.vocab.add_token_to_namespace("t", namespace='characters')

        self.oov_index = self.vocab.get_token_index('random OOV string', namespace='entities')

        self.terminal_indexers = {"entities": SingleIdTokenIndexer("entities")}
        self.nonterminal_indexers = {"rules": SingleIdTokenIndexer("rule_labels")}
        self.character_indexer = {'characters': TokenCharactersIndexer('characters')}
        def is_nonterminal(name: str) -> bool:
            if name[0] in {'<', '['}:
                return True
            return name in {'S', 'NP', 'VP'}
        self.is_nonterminal = is_nonterminal

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
                                   is_nonterminal=self.is_nonterminal)

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

    def test_as_tensor_produces_correct_output(self):
        field = self._make_field('S -> [NP, VP]')
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        assert tensor_dict.keys() == {'left', 'right'}
        assert tensor_dict['left'][0] == 'S'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.s_rule_index))
        assert tensor_dict['right'][0] == '[NP, VP]'
        assert tensor_dict['right'][1] is True
        assert tensor_dict['right'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['right'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.np_vp_index))

        field = self._make_field('NP -> test', terminal_indexers=self.character_indexer)
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        assert tensor_dict.keys() == {'left', 'right'}
        assert tensor_dict['left'][0] == 'NP'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.np_index))
        assert tensor_dict['right'][0] == 'test'
        assert tensor_dict['right'][1] is False
        assert tensor_dict['right'][2].keys() == {'characters'}
        assert_array_equal(tensor_dict['right'][2]['characters'].data.cpu().numpy(),
                           numpy.array([self.t_index, self.e_index, self.s_index, self.t_index]))

    def test_batch_tensors_does_not_modify_list(self):
        field = self._make_field('S -> [NP, VP]')
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict1 = field.as_tensor(padding_lengths)

        field = self._make_field('S -> test')
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict2 = field.as_tensor(padding_lengths)
        tensor_list = [tensor_dict1, tensor_dict2]
        assert field.batch_tensors(tensor_list) == tensor_list

    def test_doubly_nested_field_works(self):
        field1 = self._make_field('S -> [NP, VP]', terminal_indexers=self.character_indexer)
        field2 = self._make_field('NP -> cats', terminal_indexers=self.character_indexer)
        field3 = self._make_field('VP -> eat', terminal_indexers=self.character_indexer)
        list_field = ListField([ListField([field1, field2, field3]),
                                ListField([field1, field2])])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensors = list_field.as_tensor(padding_lengths)
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert isinstance(tensors[0], list)
        assert len(tensors[0]) == 3
        assert isinstance(tensors[1], list)
        assert len(tensors[1]) == 3

        tensor_dict = tensors[0][0]
        assert tensor_dict['left'][0] == 'S'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.s_rule_index))
        assert tensor_dict['right'][0] == '[NP, VP]'
        assert tensor_dict['right'][1] is True
        assert tensor_dict['right'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['right'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.np_vp_index))

        tensor_dict = tensors[0][1]
        assert tensor_dict['left'][0] == 'NP'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.np_index))
        assert tensor_dict['right'][0] == 'cats'
        assert tensor_dict['right'][1] is False
        assert tensor_dict['right'][2].keys() == {'characters'}
        assert_array_equal(tensor_dict['right'][2]['characters'].data.cpu().numpy(),
                           numpy.array([self.c_index, self.a_index, self.t_index, self.s_index]))

        tensor_dict = tensors[0][2]
        assert tensor_dict['left'][0] == 'VP'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.vp_index))
        assert tensor_dict['right'][0] == 'eat'
        assert tensor_dict['right'][1] is False
        assert tensor_dict['right'][2].keys() == {'characters'}
        assert_array_equal(tensor_dict['right'][2]['characters'].data.cpu().numpy(),
                           # Note the padding here - this is important.
                           numpy.array([self.e_index, self.a_index, self.t_index, 0]))

        tensor_dict = tensors[1][0]
        assert tensor_dict['left'][0] == 'S'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.s_rule_index))
        assert tensor_dict['right'][0] == '[NP, VP]'
        assert tensor_dict['right'][1] is True
        assert tensor_dict['right'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['right'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.np_vp_index))

        tensor_dict = tensors[1][1]
        assert tensor_dict['left'][0] == 'NP'
        assert tensor_dict['left'][1] is True
        assert tensor_dict['left'][2].keys() == {'rules'}
        assert_array_equal(tensor_dict['left'][2]['rules'].data.cpu().numpy(),
                           numpy.array(self.np_index))
        assert tensor_dict['right'][0] == 'cats'
        assert tensor_dict['right'][1] is False
        assert tensor_dict['right'][2].keys() == {'characters'}
        assert_array_equal(tensor_dict['right'][2]['characters'].data.cpu().numpy(),
                           numpy.array([self.c_index, self.a_index, self.t_index, self.s_index]))

        # This item was just padding.
        tensor_dict = tensors[1][2]
        assert tensor_dict['left'][0] == ''
        assert tensor_dict['left'][1] is True
        assert not tensor_dict['left'][2]
        assert tensor_dict['right'][0] == ''
        assert tensor_dict['right'][1] is False
        assert not tensor_dict['right'][2]
